"""
Ensemble proxy label generation.
Combines kNN + MLP (+ optional Label Propagation) predictions via soft voting or stacking.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from proxy_label_generation import (
    confusion_matrix,
    compute_weights,
    knn_query_train,
    load_dataframe,
    macro_f1,
    resolve_device,
    resolve_num_classes,
    split_indices,
    stack_embedding_column,
    to_serializable,
)
from supervised_proxy import (
    MLPClassifier,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble proxy (kNN + MLP soft voting)")
    parser.add_argument("--dataset_path", default="raw_data/tulu_300k_with_embeddings.parquet")
    parser.add_argument("--embedding_key", default="embeddings")
    parser.add_argument("--embedding_npy", default=None)
    parser.add_argument("--label_key", default="gpt_scores")
    parser.add_argument("--prediction_key", default="proxy_ensemble_label")
    parser.add_argument("--train_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--data_pool_size", type=int, default=None)

    # MLP config
    parser.add_argument("--model", default="mlp")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--mixup_alpha", type=float, default=0.0)
    parser.add_argument("--focal_loss", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--class_weight", action="store_true")
    parser.add_argument("--num_layers", type=int, default=4)

    # Feature fusion
    parser.add_argument("--use_text_features", action="store_true")
    parser.add_argument("--messages_key", default="messages")

    # kNN config
    parser.add_argument("--knn_k", type=int, default=50)
    parser.add_argument("--knn_tau", type=float, default=0.1)

    # Ensemble config
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="MLP weight in soft voting (kNN = 1-alpha)")
    parser.add_argument("--sweep_alpha", action="store_true",
                        help="Sweep alpha from 0.0 to 1.0 in 0.1 steps")

    # Label propagation
    parser.add_argument("--label_propagation", action="store_true",
                        help="Also run sklearn LabelSpreading as comparison")
    parser.add_argument("--lp_n_neighbors", type=int, default=7)
    parser.add_argument("--lp_alpha", type=float, default=0.2)
    parser.add_argument("--lp_pca_dim", type=int, default=256,
                        help="PCA dimension for label propagation (0=no PCA)")

    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default="runs/ensemble_proxy")
    parser.add_argument("--disable_tqdm", action="store_true")
    return parser.parse_args()


# ============================================================
# Probability prediction helpers
# ============================================================

def predict_probs_mlp(
    model: nn.Module,
    embeddings: torch.Tensor,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    """Return full probability matrix (N, C) from MLP."""
    model.eval()
    model.to(device)
    all_probs = []
    with torch.no_grad():
        for start in range(0, embeddings.size(0), batch_size):
            x = embeddings[start:start + batch_size].to(device)
            probs = F.softmax(model(x), dim=1)
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs).astype(np.float32)


def predict_probs_knn(
    train_labels: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_sims: np.ndarray,
    num_classes: int,
    tau: float = 0.1,
) -> np.ndarray:
    """Return kNN probability matrix (N, C) from softmax-weighted voting."""
    nbr_labels = train_labels[neighbor_indices]  # (N, K)
    weights = compute_weights(neighbor_sims, "softmax", tau)  # (N, K)
    one_hot = np.eye(num_classes, dtype=np.float64)[nbr_labels]  # (N, K, C)
    votes = (weights[:, :, None] * one_hot).sum(axis=1)  # (N, C)
    probs = votes / votes.sum(axis=1, keepdims=True)  # (N, C)
    return probs.astype(np.float32)


def evaluate_probs(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> dict:
    """Evaluate predictions from a probability matrix."""
    pred = np.argmax(probs, axis=1).astype(np.int64)
    acc = float(np.mean(pred == labels))
    conf_mat = confusion_matrix(labels, pred, num_classes)
    f1 = macro_f1(conf_mat)
    mae = float(np.mean(np.abs(pred.astype(float) - labels.astype(float))))

    per_class_acc = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            per_class_acc[f"class_{c}_acc"] = float(np.mean(pred[mask] == c))
            per_class_acc[f"class_{c}_count"] = int(mask.sum())
        else:
            per_class_acc[f"class_{c}_acc"] = 0.0
            per_class_acc[f"class_{c}_count"] = 0

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "mean_abs_error": mae,
        **per_class_acc,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    show_progress = not args.disable_tqdm

    # ── Load data ──────────────────────────────────────────────
    df = load_dataframe(args.dataset_path)
    if args.data_pool_size is not None:
        df = df.iloc[:args.data_pool_size].copy()
    df = df.reset_index(drop=True)

    labels = np.array([int(v) for v in df[args.label_key].tolist()], dtype=np.int64)
    train_idx, test_idx = split_indices(len(df), args.train_ratio, args.seed)
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    # Load embeddings
    if args.embedding_npy:
        embeddings = np.load(args.embedding_npy).astype(np.float32)
        if args.data_pool_size is not None:
            embeddings = embeddings[:args.data_pool_size]
        print(f"[info] loaded embeddings from {args.embedding_npy}: {embeddings.shape}")
    else:
        embeddings = stack_embedding_column(df, args.embedding_key, show_progress=show_progress)

    train_embeddings = embeddings[train_idx]
    test_embeddings = embeddings[test_idx]

    num_classes = resolve_num_classes(args.num_classes, train_labels, test_labels)

    print(f"[info] train={len(train_labels)}, test={len(test_labels)}, "
          f"num_classes={num_classes}, emb_dim={embeddings.shape[1]}")

    device = resolve_device(args.device)

    # ── Build MLP input features ───────────────────────────────
    if args.use_text_features:
        from feature_extraction import extract_features_batch, normalize_features
        text_feats, feat_names = extract_features_batch(df, args.messages_key, show_progress)
        train_tf, test_tf = normalize_features(text_feats[train_idx], text_feats[test_idx])
        train_input = np.concatenate([train_embeddings, train_tf], axis=1)
        test_input = np.concatenate([test_embeddings, test_tf], axis=1)
        print(f"[info] fusion: emb={embeddings.shape[1]} + text={len(feat_names)} "
              f"= {train_input.shape[1]}")
    else:
        train_input = train_embeddings
        test_input = test_embeddings

    input_dim = train_input.shape[1]

    # ── Step 1: Train MLP ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("[Step 1] Training Fusion MLP")
    print("=" * 60)

    model = MLPClassifier(input_dim, args.hidden_dim, num_classes, args.dropout)
    print(f"[info] MLP params: {sum(p.numel() for p in model.parameters()):,}")

    train_t = torch.from_numpy(train_input.astype(np.float32))
    train_labels_t = torch.from_numpy(train_labels).long()
    test_t = torch.from_numpy(test_input.astype(np.float32))
    test_labels_t = torch.from_numpy(test_labels).long()

    train_loader = DataLoader(
        TensorDataset(train_t, train_labels_t),
        batch_size=args.batch_size, shuffle=True,
    )

    class_weights = None
    if args.class_weight:
        counts = np.bincount(train_labels, minlength=num_classes).astype(np.float64)
        inv_freq = 1.0 / np.maximum(counts, 1.0)
        inv_freq = inv_freq / inv_freq.sum() * num_classes
        class_weights = torch.from_numpy(inv_freq).float()

    history = train_model(
        model, train_loader, test_t, test_labels_t,
        args, class_weights, device, num_classes, show_progress,
    )
    print(f"[MLP] best_val_acc={history['best_val_acc']:.4f}")

    # Get MLP probability predictions
    mlp_probs = predict_probs_mlp(model, test_t, device)
    mlp_metrics = evaluate_probs(mlp_probs, test_labels, num_classes)
    print(f"[MLP] accuracy={mlp_metrics['accuracy']:.4f}, "
          f"macro_f1={mlp_metrics['macro_f1']:.4f}, "
          f"mae={mlp_metrics['mean_abs_error']:.4f}")

    # ── Step 2: Run kNN ────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"[Step 2] Running kNN (K={args.knn_k}, tau={args.knn_tau})")
    print("=" * 60)

    neighbor_idx, neighbor_sim = knn_query_train(
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        knn_k=args.knn_k,
        batch_size=4096,
        device=device,
        show_progress=show_progress,
    )

    knn_probs = predict_probs_knn(
        train_labels, neighbor_idx, neighbor_sim, num_classes, args.knn_tau,
    )
    knn_metrics = evaluate_probs(knn_probs, test_labels, num_classes)
    print(f"[kNN] accuracy={knn_metrics['accuracy']:.4f}, "
          f"macro_f1={knn_metrics['macro_f1']:.4f}, "
          f"mae={knn_metrics['mean_abs_error']:.4f}")

    # ── Step 3: Soft Voting Ensemble ───────────────────────────
    print("\n" + "=" * 60)
    print("[Step 3] Soft Voting Ensemble (MLP + kNN)")
    print("=" * 60)

    if args.sweep_alpha:
        alphas = [round(a * 0.1, 1) for a in range(0, 11)]
    else:
        alphas = [args.alpha]

    alpha_results = []
    best_alpha = None
    best_acc = 0.0

    for alpha in alphas:
        ens_probs = alpha * mlp_probs + (1 - alpha) * knn_probs
        ens_metrics = evaluate_probs(ens_probs, test_labels, num_classes)
        alpha_results.append({
            "alpha": alpha,
            "accuracy": ens_metrics["accuracy"],
            "macro_f1": ens_metrics["macro_f1"],
            "mean_abs_error": ens_metrics["mean_abs_error"],
        })
        marker = ""
        if ens_metrics["accuracy"] > best_acc:
            best_acc = ens_metrics["accuracy"]
            best_alpha = alpha
            best_ens_metrics = ens_metrics
            marker = " <-- best"
        print(f"  alpha={alpha:.1f}: acc={ens_metrics['accuracy']:.4f}, "
              f"f1={ens_metrics['macro_f1']:.4f}, mae={ens_metrics['mean_abs_error']:.4f}{marker}")

    print(f"\n[Ensemble] best alpha={best_alpha}, "
          f"accuracy={best_ens_metrics['accuracy']:.4f}, "
          f"macro_f1={best_ens_metrics['macro_f1']:.4f}")

    # ── Step 4 (optional): Label Propagation ───────────────────
    lp_metrics = None
    if args.label_propagation:
        print("\n" + "=" * 60)
        print("[Step 4] Label Propagation (sklearn LabelSpreading)")
        print("=" * 60)

        import os
        # Limit threads to avoid OpenBLAS segfaults
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
        os.environ.setdefault("OMP_NUM_THREADS", "8")

        try:
            from sklearn.semi_supervised import LabelSpreading

            # Prepare: use raw embeddings for LP (not fusion features)
            all_embeddings_lp = embeddings  # (N, D)

            # PCA dimensionality reduction (required for high-dim)
            if args.lp_pca_dim > 0 and all_embeddings_lp.shape[1] > args.lp_pca_dim:
                from sklearn.decomposition import PCA
                print(f"[LP] PCA {all_embeddings_lp.shape[1]} -> {args.lp_pca_dim}")
                pca = PCA(n_components=args.lp_pca_dim, random_state=args.seed)
                all_embeddings_pca = pca.fit_transform(all_embeddings_lp).astype(np.float32)
                explained = pca.explained_variance_ratio_.sum()
                print(f"[LP] PCA explained variance: {explained:.4f}")
            else:
                all_embeddings_pca = all_embeddings_lp

            # Labels: train = known, test = -1
            all_labels_lp = np.full(len(labels), -1, dtype=np.int64)
            all_labels_lp[train_idx] = train_labels

            print(f"[LP] n_neighbors={args.lp_n_neighbors}, alpha={args.lp_alpha}")
            print(f"[LP] labeled={len(train_idx)}, unlabeled={len(test_idx)}")

            ls = LabelSpreading(
                kernel="knn",
                n_neighbors=args.lp_n_neighbors,
                alpha=args.lp_alpha,
                max_iter=50,
                n_jobs=4,
            )
            ls.fit(all_embeddings_pca, all_labels_lp)

            # Evaluate on test set
            lp_pred = ls.transduction_[test_idx]
            lp_acc = float(np.mean(lp_pred == test_labels))
            conf_mat_lp = confusion_matrix(test_labels, lp_pred, num_classes)
            lp_f1 = macro_f1(conf_mat_lp)
            lp_mae = float(np.mean(np.abs(lp_pred.astype(float) - test_labels.astype(float))))

            lp_metrics = {"accuracy": lp_acc, "macro_f1": lp_f1, "mean_abs_error": lp_mae}

            # Per-class accuracy
            for c in range(num_classes):
                mask = test_labels == c
                if mask.sum() > 0:
                    lp_metrics[f"class_{c}_acc"] = float(np.mean(lp_pred[mask] == c))

            print(f"[LP] accuracy={lp_acc:.4f}, macro_f1={lp_f1:.4f}, mae={lp_mae:.4f}")

            # Also try LP + MLP ensemble
            lp_probs_raw = ls.label_distributions_[test_idx]
            lp_probs_norm = lp_probs_raw / lp_probs_raw.sum(axis=1, keepdims=True)
            for alpha_lp in [0.7, 0.8, 0.9]:
                ens_lp = alpha_lp * mlp_probs + (1 - alpha_lp) * lp_probs_norm.astype(np.float32)
                ens_lp_metrics = evaluate_probs(ens_lp, test_labels, num_classes)
                print(f"  MLP+LP alpha={alpha_lp:.1f}: acc={ens_lp_metrics['accuracy']:.4f}, "
                      f"f1={ens_lp_metrics['macro_f1']:.4f}")

        except Exception as e:
            print(f"[LP] FAILED: {e}")
            print("[LP] Skipping label propagation (may need more memory or fewer threads)")

    # ── Save results ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[Summary]")
    print("=" * 60)
    print(f"  MLP only:  acc={mlp_metrics['accuracy']:.4f}, f1={mlp_metrics['macro_f1']:.4f}")
    print(f"  kNN only:  acc={knn_metrics['accuracy']:.4f}, f1={knn_metrics['macro_f1']:.4f}")
    print(f"  Ensemble:  acc={best_ens_metrics['accuracy']:.4f}, "
          f"f1={best_ens_metrics['macro_f1']:.4f} (alpha={best_alpha})")
    if lp_metrics:
        print(f"  LabelProp: acc={lp_metrics['accuracy']:.4f}, f1={lp_metrics['macro_f1']:.4f}")

    # Check error correlation
    mlp_pred = np.argmax(mlp_probs, axis=1)
    knn_pred = np.argmax(knn_probs, axis=1)
    mlp_correct = (mlp_pred == test_labels)
    knn_correct = (knn_pred == test_labels)
    both_correct = (mlp_correct & knn_correct).sum()
    either_correct = (mlp_correct | knn_correct).sum()
    disagreement = (mlp_pred != knn_pred).sum()
    print(f"\n  Error analysis:")
    print(f"    MLP correct: {mlp_correct.sum()} ({mlp_correct.mean()*100:.1f}%)")
    print(f"    kNN correct: {knn_correct.sum()} ({knn_correct.mean()*100:.1f}%)")
    print(f"    Both correct: {both_correct} ({both_correct/len(test_labels)*100:.1f}%)")
    print(f"    Either correct: {either_correct} ({either_correct/len(test_labels)*100:.1f}%)")
    print(f"    Disagreement: {disagreement} ({disagreement/len(test_labels)*100:.1f}%)")

    # Save JSON outputs
    results = {
        "train_ratio": args.train_ratio,
        "train_rows": len(train_labels),
        "test_rows": len(test_labels),
        "num_classes": num_classes,
        "knn_k": args.knn_k,
        "knn_tau": args.knn_tau,
        "best_alpha": best_alpha,
        "mlp_metrics": mlp_metrics,
        "knn_metrics": knn_metrics,
        "ensemble_metrics": best_ens_metrics,
        "disagreement_rate": float(disagreement / len(test_labels)),
        "either_correct_rate": float(either_correct / len(test_labels)),
    }
    if lp_metrics:
        results["lp_metrics"] = lp_metrics

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(to_serializable(results), f, indent=2)

    with open(output_dir / "alpha_sweep.json", "w") as f:
        json.dump(to_serializable(alpha_results), f, indent=2)

    print(f"\n[info] results saved to {output_dir}")


if __name__ == "__main__":
    main()
