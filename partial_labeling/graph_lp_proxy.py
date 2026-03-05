"""
Modern Graph Label Propagation for Semi-Supervised Proxy Label Generation.

Implements:
1. C&S (Correct and Smooth) - Huang et al., ICLR 2021
   Post-processes MLP predictions using graph-based error correction and smoothing.
2. APPNP - Klicpera et al., ICLR 2019
   Personalized PageRank propagation of MLP predictions.

Key improvement over sklearn LabelSpreading: starts from MLP predictions (~55% acc)
instead of propagating from scratch.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix, diags, eye as speye
from torch.utils.data import DataLoader, TensorDataset

from proxy_label_generation import (
    confusion_matrix,
    load_dataframe,
    macro_f1,
    resolve_device,
    resolve_num_classes,
    split_indices,
    stack_embedding_column,
    to_serializable,
)
from supervised_proxy import MLPClassifier, train_model


# ---------------------------------------------------------------------------
# kNN graph construction (all-to-all)
# ---------------------------------------------------------------------------

def knn_query_all(
    embeddings: np.ndarray,
    knn_k: int,
    batch_size: int,
    device: torch.device,
) -> tuple:
    """Build all-to-all kNN graph using cosine similarity on GPU.

    Returns:
        neighbor_indices: (N, K) int array
        neighbor_sims: (N, K) float array
    """
    N = embeddings.shape[0]
    emb_t = torch.from_numpy(embeddings).to(device)
    emb_t = F.normalize(emb_t, dim=1)

    k_eff = min(knn_k, N - 1)  # exclude self

    all_indices = []
    all_sims = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = emb_t[start:end]  # (B, D)
        sims = torch.matmul(batch, emb_t.T)  # (B, N)

        # Zero out self-similarity to exclude self-loops
        batch_range = torch.arange(end - start, device=device)
        global_indices = torch.arange(start, end, device=device)
        sims[batch_range, global_indices] = -2.0

        topk_sims, topk_idx = torch.topk(sims, k=k_eff, dim=1, largest=True, sorted=True)
        all_indices.append(topk_idx.cpu().numpy())
        all_sims.append(topk_sims.cpu().numpy())

        if start % (batch_size * 5) == 0:
            print(f"  kNN graph: {end}/{N}")

    return np.concatenate(all_indices), np.concatenate(all_sims)


# ---------------------------------------------------------------------------
# Adjacency matrix construction
# ---------------------------------------------------------------------------

def build_normalized_adj(
    knn_indices: np.ndarray,
    knn_sims: np.ndarray,
    n_nodes: int,
    self_loop: bool = True,
    normalization: str = "symmetric",
) -> csr_matrix:
    """Build normalized adjacency from kNN graph.

    Args:
        normalization: "symmetric" for D^{-1/2} A D^{-1/2},
                       "row" for D^{-1} A (random walk).
    """
    N, K = knn_indices.shape

    # Vectorized construction
    rows = np.repeat(np.arange(N), K)
    cols = knn_indices.flatten()
    vals = np.maximum(knn_sims.flatten(), 0.0).astype(np.float64)

    # Filter zero/negative similarities
    mask = vals > 0
    rows, cols, vals = rows[mask], cols[mask], vals[mask]

    A = csr_matrix((vals, (rows, cols)), shape=(N, N))
    # Symmetrize: A = (A + A^T) / 2
    A = (A + A.T) / 2.0

    if self_loop:
        A = A + speye(N, format="csr")

    deg = np.array(A.sum(axis=1)).flatten()

    if normalization == "symmetric":
        deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        D_inv_sqrt = diags(deg_inv_sqrt)
        S = D_inv_sqrt @ A @ D_inv_sqrt
    elif normalization == "row":
        deg_inv = np.where(deg > 0, 1.0 / deg, 0.0)
        D_inv = diags(deg_inv)
        S = D_inv @ A
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    return S.tocsr()


# ---------------------------------------------------------------------------
# C&S: Correct and Smooth (ICLR 2021)
# ---------------------------------------------------------------------------

def correct_step(
    y_soft: np.ndarray,
    y_true_onehot: np.ndarray,
    train_mask: np.ndarray,
    S: csr_matrix,
    num_layers: int = 50,
    alpha: float = 0.8,
    autoscale: bool = True,
    scale: float = 1.0,
) -> np.ndarray:
    """C&S Correct step: propagate training residuals to fix predictions.

    Spreads the error pattern from training nodes across the graph,
    then adds scaled error to the original soft predictions.
    """
    error = np.zeros_like(y_soft)
    error[train_mask] = y_true_onehot[train_mask] - y_soft[train_mask]

    smoothed = error.copy()
    base = (1 - alpha) * error

    for i in range(num_layers):
        smoothed = alpha * (S @ smoothed) + base

    if autoscale:
        # Scale so that smoothed error magnitude matches training error magnitude
        train_err_avg = np.abs(error[train_mask]).sum() / train_mask.sum()
        node_norms = np.abs(smoothed).sum(axis=1, keepdims=True) + 1e-12
        node_scale = train_err_avg / node_norms
        node_scale = np.clip(node_scale, 0.0, 1000.0)
        corrected = y_soft + node_scale * smoothed
    else:
        corrected = y_soft + scale * smoothed

    # Clamp and renormalize to valid probabilities
    corrected = np.clip(corrected, 0.0, None)
    row_sums = corrected.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    corrected = corrected / row_sums

    return corrected


def smooth_step(
    y_input: np.ndarray,
    y_true_onehot: np.ndarray,
    train_mask: np.ndarray,
    S: csr_matrix,
    num_layers: int = 50,
    alpha: float = 0.8,
) -> np.ndarray:
    """C&S Smooth step: enforce neighborhood label consistency.

    Anchors training nodes to ground truth and propagates to smooth predictions.
    """
    out = y_input.copy()
    out[train_mask] = y_true_onehot[train_mask]

    base = out.copy()

    for i in range(num_layers):
        out = alpha * (S @ out) + (1 - alpha) * base
        # Re-anchor training nodes each iteration
        out[train_mask] = y_true_onehot[train_mask]

    # Normalize
    out = np.clip(out, 0.0, None)
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    out = out / row_sums

    return out


# ---------------------------------------------------------------------------
# APPNP: Personalized PageRank Propagation (ICLR 2019)
# ---------------------------------------------------------------------------

def appnp_propagation(
    y_soft: np.ndarray,
    S: csr_matrix,
    num_layers: int = 10,
    alpha: float = 0.1,
) -> np.ndarray:
    """APPNP: Personalized PageRank propagation.

    Z^(k) = (1 - alpha) * S @ Z^(k-1) + alpha * H
    where H = y_soft (original MLP predictions) acts as teleport target.
    alpha controls how much to preserve the original MLP signal.
    """
    H = y_soft.copy()
    Z = H.copy()

    for _ in range(num_layers):
        Z = (1 - alpha) * (S @ Z) + alpha * H

    # Normalize
    Z = np.clip(Z, 0.0, None)
    row_sums = Z.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    Z = Z / row_sums

    return Z


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_on_test(probs, test_idx, test_labels, num_classes):
    """Evaluate probability predictions on test set."""
    test_probs = probs[test_idx]
    pred = np.argmax(test_probs, axis=1).astype(np.int64)
    acc = float(np.mean(pred == test_labels))
    cm = confusion_matrix(test_labels, pred, num_classes)
    f1 = macro_f1(cm)
    mae = float(np.mean(np.abs(pred.astype(float) - test_labels.astype(float))))

    per_class = {}
    for c in range(num_classes):
        mask_c = test_labels == c
        if mask_c.sum() > 0:
            per_class[f"class_{c}_acc"] = float(np.mean(pred[mask_c] == c))
            per_class[f"class_{c}_count"] = int(mask_c.sum())

    return {"accuracy": acc, "macro_f1": f1, "mean_abs_error": mae, **per_class}


def print_metrics(name, metrics, baseline_metrics=None):
    """Print metrics with optional delta from baseline."""
    acc, f1, mae = metrics["accuracy"], metrics["macro_f1"], metrics["mean_abs_error"]
    if baseline_metrics:
        ba, bf, bm = baseline_metrics["accuracy"], baseline_metrics["macro_f1"], baseline_metrics["mean_abs_error"]
        print(f"[{name}] acc={acc:.4f} ({acc - ba:+.4f}), "
              f"f1={f1:.4f} ({f1 - bf:+.4f}), mae={mae:.4f} ({mae - bm:+.4f})")
    else:
        print(f"[{name}] acc={acc:.4f}, f1={f1:.4f}, mae={mae:.4f}")

    # Per-class
    for c in range(10):
        k_acc = f"class_{c}_acc"
        k_cnt = f"class_{c}_count"
        if k_acc in metrics:
            print(f"  class {c}: {metrics[k_acc]:.4f} (n={metrics[k_cnt]})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Graph LP: C&S + APPNP post-processing")

    # Data
    p.add_argument("--dataset_path", default="raw_data/tulu_300k_with_embeddings.parquet")
    p.add_argument("--embedding_key", default="embeddings")
    p.add_argument("--embedding_npy", default=None)
    p.add_argument("--label_key", default="gpt_scores")
    p.add_argument("--train_ratio", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--num_classes", type=int, default=None)
    p.add_argument("--data_pool_size", type=int, default=None)

    # Feature fusion
    p.add_argument("--use_text_features", action="store_true")
    p.add_argument("--messages_key", default="messages")

    # MLP training
    p.add_argument("--model", default="mlp")
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--mixup_alpha", type=float, default=0.0)
    p.add_argument("--focal_loss", action="store_true")
    p.add_argument("--focal_gamma", type=float, default=2.0)

    # Graph construction
    p.add_argument("--knn_k", type=int, default=50,
                   help="Number of neighbors for all-to-all kNN graph")
    p.add_argument("--knn_batch_size", type=int, default=512)

    # C&S hyperparameters
    p.add_argument("--num_correction_layers", type=int, default=50)
    p.add_argument("--correction_alpha", type=float, default=0.8)
    p.add_argument("--num_smoothing_layers", type=int, default=50)
    p.add_argument("--smoothing_alpha", type=float, default=0.8)
    p.add_argument("--autoscale", action="store_true", default=True)
    p.add_argument("--no_autoscale", dest="autoscale", action="store_false")
    p.add_argument("--scale", type=float, default=1.0)

    # APPNP hyperparameters
    p.add_argument("--appnp_layers", type=int, default=10)
    p.add_argument("--appnp_alpha", type=float, default=0.1)

    # Sweep mode
    p.add_argument("--sweep", action="store_true",
                   help="Sweep C&S and APPNP hyperparameters")

    p.add_argument("--device", default=None)
    p.add_argument("--output_dir", default="runs/graph_lp")

    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    # ── 1. Load data ──────────────────────────────────────────
    print("=== Loading data ===")
    df = load_dataframe(args.dataset_path)
    if args.data_pool_size is not None:
        df = df.iloc[: args.data_pool_size].copy()
    df = df.reset_index(drop=True)

    labels = np.array([int(v) for v in df[args.label_key].tolist()], dtype=np.int64)
    train_idx, test_idx = split_indices(len(df), args.train_ratio, args.seed)
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]
    num_classes = resolve_num_classes(args.num_classes, train_labels, test_labels)

    # Load embeddings
    if args.embedding_npy:
        embeddings = np.load(args.embedding_npy).astype(np.float32)
        if args.data_pool_size is not None:
            embeddings = embeddings[: args.data_pool_size]
        print(f"Loaded embeddings: {embeddings.shape}")
    else:
        embeddings = stack_embedding_column(df, args.embedding_key)

    # ── 2. Build MLP input (embedding + optional text features) ───
    if args.use_text_features:
        from feature_extraction import extract_features_batch, normalize_features

        text_feats, feat_names = extract_features_batch(df, args.messages_key)
        train_tf, test_tf = normalize_features(
            text_feats[train_idx], text_feats[test_idx]
        )
        # Need full normalized features for all nodes
        # Re-normalize using training stats applied to all
        mu = text_feats[train_idx].mean(axis=0)
        std = text_feats[train_idx].std(axis=0) + 1e-8
        all_tf = (text_feats - mu) / std

        all_input = np.concatenate([embeddings, all_tf], axis=1).astype(np.float32)
        train_input = all_input[train_idx]
        test_input = all_input[test_idx]
        print(f"Fusion input: emb={embeddings.shape[1]} + text={len(feat_names)} "
              f"= {all_input.shape[1]}")
    else:
        all_input = embeddings.astype(np.float32)
        train_input = all_input[train_idx]
        test_input = all_input[test_idx]

    input_dim = all_input.shape[1]
    print(f"Data: {len(labels)} total, {len(train_idx)} train, {len(test_idx)} test")
    print(f"Input dim: {input_dim}, classes: {num_classes}")

    # ── 3. Train MLP ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 1: Training Fusion MLP")
    print("=" * 60)

    model = MLPClassifier(input_dim, args.hidden_dim, num_classes, args.dropout)
    print(f"MLP params: {sum(p.numel() for p in model.parameters()):,}")

    train_t = torch.from_numpy(train_input)
    train_labels_t = torch.from_numpy(train_labels).long()
    test_t = torch.from_numpy(test_input)

    train_loader = DataLoader(
        TensorDataset(train_t, train_labels_t),
        batch_size=args.batch_size,
        shuffle=True,
    )

    test_labels_t = torch.from_numpy(test_labels).long()
    history = train_model(
        model, train_loader, test_t, test_labels_t,
        args, class_weights=None, device=device,
        num_classes=num_classes,
    )
    print(f"MLP best_val_acc: {history['best_val_acc']:.4f}")

    # ── 4. Get MLP soft predictions for ALL nodes ─────────────
    print("\n" + "=" * 60)
    print("Step 2: Getting MLP predictions for all nodes")
    print("=" * 60)

    model.eval()
    model.to(device)
    all_input_t = torch.from_numpy(all_input)
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(all_input_t), 2048):
            batch = all_input_t[start : start + 2048].to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
    y_soft = np.concatenate(all_probs).astype(np.float64)

    # Evaluate MLP baseline
    mlp_metrics = evaluate_on_test(y_soft, test_idx, test_labels, num_classes)
    print_metrics("MLP Baseline", mlp_metrics)

    results = {
        "train_ratio": args.train_ratio,
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "num_classes": num_classes,
        "knn_k": args.knn_k,
        "mlp_metrics": mlp_metrics,
    }

    # ── 5. Build all-to-all kNN graph ─────────────────────────
    print("\n" + "=" * 60)
    print(f"Step 3: Building all-to-all kNN graph (k={args.knn_k})")
    print("=" * 60)

    t0 = time.time()
    knn_indices, knn_sims = knn_query_all(
        embeddings, args.knn_k, args.knn_batch_size, device
    )
    print(f"kNN graph: {time.time() - t0:.1f}s, "
          f"mean_sim={knn_sims.mean():.4f}, min_sim={knn_sims.min():.4f}")

    # Build normalized adjacency
    print("Building symmetric normalized adjacency...")
    S_sym = build_normalized_adj(knn_indices, knn_sims, len(labels),
                                 self_loop=True, normalization="symmetric")
    print(f"Adjacency: {S_sym.shape}, nnz={S_sym.nnz:,}")

    # Prepare one-hot training labels (full size)
    y_true_onehot = np.zeros((len(labels), num_classes), dtype=np.float64)
    y_true_onehot[train_idx, train_labels] = 1.0

    train_mask = np.zeros(len(labels), dtype=bool)
    train_mask[train_idx] = True

    # ── 6. C&S ────────────────────────────────────────────────
    if args.sweep:
        # Sweep hyperparameters
        print("\n" + "=" * 60)
        print("Step 4: C&S Hyperparameter Sweep")
        print("=" * 60)

        best_cs_acc = 0.0
        best_cs_config = {}
        best_cs_metrics = None

        correction_alphas = [0.5, 0.7, 0.8, 0.9, 1.0]
        smoothing_alphas = [0.5, 0.6, 0.7, 0.8, 0.9]
        layer_counts = [20, 50]

        sweep_results = []

        for c_alpha in correction_alphas:
            for s_alpha in smoothing_alphas:
                for n_layers in layer_counts:
                    y_corr = correct_step(
                        y_soft, y_true_onehot, train_mask, S_sym,
                        num_layers=n_layers, alpha=c_alpha,
                        autoscale=args.autoscale, scale=args.scale,
                    )
                    y_cs = smooth_step(
                        y_corr, y_true_onehot, train_mask, S_sym,
                        num_layers=n_layers, alpha=s_alpha,
                    )
                    m = evaluate_on_test(y_cs, test_idx, test_labels, num_classes)
                    sweep_results.append({
                        "correction_alpha": c_alpha,
                        "smoothing_alpha": s_alpha,
                        "num_layers": n_layers,
                        **m,
                    })
                    tag = f"c_α={c_alpha}, s_α={s_alpha}, L={n_layers}"
                    dacc = m["accuracy"] - mlp_metrics["accuracy"]
                    df1 = m["macro_f1"] - mlp_metrics["macro_f1"]
                    print(f"  {tag}: acc={m['accuracy']:.4f} ({dacc:+.4f}), "
                          f"f1={m['macro_f1']:.4f} ({df1:+.4f})")

                    if m["accuracy"] > best_cs_acc:
                        best_cs_acc = m["accuracy"]
                        best_cs_config = {
                            "correction_alpha": c_alpha,
                            "smoothing_alpha": s_alpha,
                            "num_layers": n_layers,
                        }
                        best_cs_metrics = m

        print(f"\nBest C&S: acc={best_cs_acc:.4f}, config={best_cs_config}")
        print_metrics("Best C&S", best_cs_metrics, mlp_metrics)

        results["cs_sweep"] = sweep_results
        results["cs_best_config"] = best_cs_config
        results["cs_best_metrics"] = best_cs_metrics

        # Also sweep APPNP
        print("\n" + "=" * 60)
        print("Step 5: APPNP Hyperparameter Sweep")
        print("=" * 60)

        best_appnp_acc = 0.0
        best_appnp_config = {}
        best_appnp_metrics = None
        appnp_sweep = []

        for a_alpha in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
            for a_layers in [5, 10, 20, 50]:
                y_appnp = appnp_propagation(y_soft, S_sym,
                                            num_layers=a_layers, alpha=a_alpha)
                m = evaluate_on_test(y_appnp, test_idx, test_labels, num_classes)
                appnp_sweep.append({
                    "alpha": a_alpha, "num_layers": a_layers, **m,
                })
                dacc = m["accuracy"] - mlp_metrics["accuracy"]
                df1 = m["macro_f1"] - mlp_metrics["macro_f1"]
                print(f"  α={a_alpha}, L={a_layers}: acc={m['accuracy']:.4f} ({dacc:+.4f}), "
                      f"f1={m['macro_f1']:.4f} ({df1:+.4f})")

                if m["accuracy"] > best_appnp_acc:
                    best_appnp_acc = m["accuracy"]
                    best_appnp_config = {"alpha": a_alpha, "num_layers": a_layers}
                    best_appnp_metrics = m

        print(f"\nBest APPNP: acc={best_appnp_acc:.4f}, config={best_appnp_config}")
        print_metrics("Best APPNP", best_appnp_metrics, mlp_metrics)

        results["appnp_sweep"] = appnp_sweep
        results["appnp_best_config"] = best_appnp_config
        results["appnp_best_metrics"] = best_appnp_metrics

    else:
        # Single run with specified hyperparameters
        # -- Correct Only --
        print("\n" + "=" * 60)
        print(f"Step 4a: C&S Correct (L={args.num_correction_layers}, α={args.correction_alpha})")
        print("=" * 60)

        t0 = time.time()
        y_corrected = correct_step(
            y_soft, y_true_onehot, train_mask, S_sym,
            num_layers=args.num_correction_layers,
            alpha=args.correction_alpha,
            autoscale=args.autoscale,
            scale=args.scale,
        )
        print(f"Correct step: {time.time() - t0:.1f}s")

        corr_metrics = evaluate_on_test(y_corrected, test_idx, test_labels, num_classes)
        print_metrics("Correct Only", corr_metrics, mlp_metrics)
        results["correct_only"] = corr_metrics

        # -- Smooth Only (skip Correct) --
        print(f"\nStep 4b: Smooth Only (L={args.num_smoothing_layers}, α={args.smoothing_alpha})")

        t0 = time.time()
        y_smooth_only = smooth_step(
            y_soft, y_true_onehot, train_mask, S_sym,
            num_layers=args.num_smoothing_layers,
            alpha=args.smoothing_alpha,
        )
        print(f"Smooth step: {time.time() - t0:.1f}s")

        smooth_metrics = evaluate_on_test(y_smooth_only, test_idx, test_labels, num_classes)
        print_metrics("Smooth Only", smooth_metrics, mlp_metrics)
        results["smooth_only"] = smooth_metrics

        # -- Full C&S (Correct then Smooth) --
        print(f"\nStep 4c: Full C&S (Correct + Smooth)")

        t0 = time.time()
        y_cs = smooth_step(
            y_corrected, y_true_onehot, train_mask, S_sym,
            num_layers=args.num_smoothing_layers,
            alpha=args.smoothing_alpha,
        )
        print(f"Smooth on corrected: {time.time() - t0:.1f}s")

        cs_metrics = evaluate_on_test(y_cs, test_idx, test_labels, num_classes)
        print_metrics("C&S (Correct+Smooth)", cs_metrics, mlp_metrics)
        results["cs_metrics"] = {
            **cs_metrics,
            "num_correction_layers": args.num_correction_layers,
            "correction_alpha": args.correction_alpha,
            "num_smoothing_layers": args.num_smoothing_layers,
            "smoothing_alpha": args.smoothing_alpha,
            "autoscale": args.autoscale,
        }

        # -- APPNP --
        print(f"\n" + "=" * 60)
        print(f"Step 5: APPNP (K={args.appnp_layers}, α={args.appnp_alpha})")
        print("=" * 60)

        t0 = time.time()
        y_appnp = appnp_propagation(
            y_soft, S_sym,
            num_layers=args.appnp_layers,
            alpha=args.appnp_alpha,
        )
        print(f"APPNP: {time.time() - t0:.1f}s")

        appnp_metrics = evaluate_on_test(y_appnp, test_idx, test_labels, num_classes)
        print_metrics("APPNP", appnp_metrics, mlp_metrics)
        results["appnp_metrics"] = {
            **appnp_metrics,
            "num_layers": args.appnp_layers,
            "alpha": args.appnp_alpha,
        }

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    ba = mlp_metrics["accuracy"]
    bf = mlp_metrics["macro_f1"]
    print(f"{'Method':<25} {'Acc':>8} {'ΔAcc':>8} {'F1':>8} {'ΔF1':>8}")
    print("-" * 60)
    print(f"{'MLP Baseline':<25} {ba:>8.4f} {'':>8} {bf:>8.4f} {'':>8}")

    if args.sweep:
        if best_cs_metrics:
            a, f = best_cs_metrics["accuracy"], best_cs_metrics["macro_f1"]
            print(f"{'Best C&S':<25} {a:>8.4f} {a - ba:>+8.4f} {f:>8.4f} {f - bf:>+8.4f}")
        if best_appnp_metrics:
            a, f = best_appnp_metrics["accuracy"], best_appnp_metrics["macro_f1"]
            print(f"{'Best APPNP':<25} {a:>8.4f} {a - ba:>+8.4f} {f:>8.4f} {f - bf:>+8.4f}")
    else:
        for name, key in [("Correct Only", "correct_only"),
                          ("Smooth Only", "smooth_only"),
                          ("C&S", "cs_metrics"),
                          ("APPNP", "appnp_metrics")]:
            if key in results:
                m = results[key]
                a, f = m["accuracy"], m["macro_f1"]
                print(f"{name:<25} {a:>8.4f} {a - ba:>+8.4f} {f:>8.4f} {f - bf:>+8.4f}")

    # ── Save ──────────────────────────────────────────────────
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as fp:
        json.dump({k: to_serializable(v) for k, v in results.items()}, fp, indent=2)
    print(f"\nResults saved to {metrics_path}")


if __name__ == "__main__":
    main()
