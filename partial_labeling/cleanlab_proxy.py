"""
Cleanlab-based label noise detection and cleaning for proxy label generation.
Uses cross-validated predicted probabilities to identify noisy training labels,
then retrains on cleaned data.
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
    predict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cleanlab label noise detection + cleaning")
    parser.add_argument("--dataset_path", default="raw_data/tulu_300k_with_embeddings.parquet")
    parser.add_argument("--embedding_key", default="embeddings")
    parser.add_argument("--embedding_npy", default=None)
    parser.add_argument("--label_key", default="gpt_scores")
    parser.add_argument("--prediction_key", default="proxy_cleanlab_label")
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

    # Cleanlab config
    parser.add_argument("--cv_folds", type=int, default=5,
                        help="Number of cross-validation folds for out-of-sample predictions")
    parser.add_argument("--clean_method", choices=["remove", "relabel", "none"], default="remove",
                        help="How to handle detected label issues: "
                             "remove=drop noisy samples, "
                             "relabel=replace with model's prediction, "
                             "none=only diagnose")

    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default="runs/cleanlab_proxy")
    parser.add_argument("--disable_tqdm", action="store_true")
    return parser.parse_args()


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


def evaluate(pred: np.ndarray, labels: np.ndarray, num_classes: int) -> dict:
    """Compute accuracy, macro F1, MAE, per-class accuracy."""
    acc = float(np.mean(pred == labels))
    conf_mat = confusion_matrix(labels, pred, num_classes)
    f1 = macro_f1(conf_mat)
    mae = float(np.mean(np.abs(pred.astype(float) - labels.astype(float))))

    result = {"accuracy": acc, "macro_f1": f1, "mean_abs_error": mae}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            result[f"class_{c}_acc"] = float(np.mean(pred[mask] == c))
            result[f"class_{c}_count"] = int(mask.sum())
    return result


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

    num_classes = resolve_num_classes(args.num_classes, train_labels, test_labels)

    # Build input features
    if args.use_text_features:
        from feature_extraction import extract_features_batch, normalize_features
        text_feats, feat_names = extract_features_batch(df, args.messages_key, show_progress)
        train_tf, test_tf = normalize_features(text_feats[train_idx], text_feats[test_idx])
        train_input = np.concatenate([embeddings[train_idx], train_tf], axis=1)
        test_input = np.concatenate([embeddings[test_idx], test_tf], axis=1)
        print(f"[info] fusion: emb={embeddings.shape[1]} + text={len(feat_names)} "
              f"= {train_input.shape[1]}")
    else:
        train_input = embeddings[train_idx]
        test_input = embeddings[test_idx]

    input_dim = train_input.shape[1]
    device = resolve_device(args.device)

    print(f"[info] train={len(train_labels)}, test={len(test_labels)}, "
          f"num_classes={num_classes}")

    # ── Step 1: Baseline (train on all labels) ─────────────────
    print("\n" + "=" * 60)
    print("[Step 1] Baseline: Train on all labels")
    print("=" * 60)

    model_base = MLPClassifier(input_dim, args.hidden_dim, num_classes, args.dropout)
    train_t = torch.from_numpy(train_input.astype(np.float32))
    train_labels_t = torch.from_numpy(train_labels).long()
    test_t = torch.from_numpy(test_input.astype(np.float32))
    test_labels_t = torch.from_numpy(test_labels).long()

    train_loader = DataLoader(
        TensorDataset(train_t, train_labels_t),
        batch_size=args.batch_size, shuffle=True,
    )
    history_base = train_model(
        model_base, train_loader, test_t, test_labels_t,
        args, None, device, num_classes, show_progress,
    )

    base_pred, _ = predict(model_base, test_t, device)
    base_metrics = evaluate(base_pred, test_labels, num_classes)
    print(f"[Baseline] accuracy={base_metrics['accuracy']:.4f}, "
          f"macro_f1={base_metrics['macro_f1']:.4f}, "
          f"mae={base_metrics['mean_abs_error']:.4f}")

    # ── Step 2: Cross-validated out-of-sample predictions ──────
    print("\n" + "=" * 60)
    print(f"[Step 2] {args.cv_folds}-fold CV for out-of-sample predictions")
    print("=" * 60)

    from sklearn.model_selection import StratifiedKFold

    cv_pred_probs = np.zeros((len(train_labels), num_classes), dtype=np.float32)
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    for fold_i, (fold_train, fold_val) in enumerate(skf.split(train_input, train_labels)):
        print(f"\n  Fold {fold_i + 1}/{args.cv_folds}: "
              f"train={len(fold_train)}, val={len(fold_val)}")

        fold_model = MLPClassifier(input_dim, args.hidden_dim, num_classes, args.dropout)
        fold_train_t = torch.from_numpy(train_input[fold_train].astype(np.float32))
        fold_train_labels_t = torch.from_numpy(train_labels[fold_train]).long()
        fold_val_t = torch.from_numpy(train_input[fold_val].astype(np.float32))
        fold_val_labels_t = torch.from_numpy(train_labels[fold_val]).long()

        fold_loader = DataLoader(
            TensorDataset(fold_train_t, fold_train_labels_t),
            batch_size=args.batch_size, shuffle=True,
        )

        train_model(
            fold_model, fold_loader, fold_val_t, fold_val_labels_t,
            args, None, device, num_classes, show_progress=False,
        )

        fold_probs = predict_probs_mlp(fold_model, fold_val_t, device)
        cv_pred_probs[fold_val] = fold_probs

        fold_pred = np.argmax(fold_probs, axis=1)
        fold_acc = float(np.mean(fold_pred == train_labels[fold_val]))
        print(f"  Fold {fold_i + 1} val accuracy: {fold_acc:.4f}")

    cv_pred = np.argmax(cv_pred_probs, axis=1)
    cv_acc = float(np.mean(cv_pred == train_labels))
    print(f"\n[CV] overall OOS accuracy: {cv_acc:.4f}")

    # ── Step 3: Cleanlab noise detection ───────────────────────
    print("\n" + "=" * 60)
    print("[Step 3] Cleanlab noise detection")
    print("=" * 60)

    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores

    # Find label issues
    label_issues_mask = find_label_issues(
        labels=train_labels,
        pred_probs=cv_pred_probs,
        return_indices_ranked_by=None,
    )
    label_issues_idx = find_label_issues(
        labels=train_labels,
        pred_probs=cv_pred_probs,
        return_indices_ranked_by="self_confidence",
    )

    n_issues = label_issues_mask.sum()
    issue_rate = n_issues / len(train_labels)
    print(f"[Cleanlab] detected {n_issues} label issues ({issue_rate*100:.1f}% of training data)")

    # Get label quality scores
    quality_scores = get_label_quality_scores(
        labels=train_labels,
        pred_probs=cv_pred_probs,
    )

    # Analyze issues by class
    print(f"\n[Cleanlab] Issues per class:")
    for c in range(num_classes):
        c_mask = train_labels == c
        c_issues = label_issues_mask[c_mask].sum()
        c_total = c_mask.sum()
        c_rate = c_issues / c_total if c_total > 0 else 0
        print(f"  class {c}: {c_issues}/{c_total} issues ({c_rate*100:.1f}%)")

    # Show top-K most suspicious samples
    print(f"\n[Cleanlab] Top-20 most suspicious labels (lowest quality score):")
    sorted_by_quality = np.argsort(quality_scores)
    for rank, idx in enumerate(sorted_by_quality[:20]):
        given = train_labels[idx]
        suggested = cv_pred[idx]
        score = quality_scores[idx]
        print(f"  #{rank+1}: idx={idx}, given_label={given}, "
              f"model_pred={suggested}, quality={score:.4f}")

    # Build noise analysis
    noise_analysis = {
        "total_train": int(len(train_labels)),
        "label_issues_count": int(n_issues),
        "label_issues_rate": float(issue_rate),
        "cv_oos_accuracy": float(cv_acc),
        "issues_per_class": {},
        "top_suspicious": [],
    }
    for c in range(num_classes):
        c_mask = train_labels == c
        c_issues = int(label_issues_mask[c_mask].sum())
        c_total = int(c_mask.sum())
        noise_analysis["issues_per_class"][str(c)] = {
            "issues": c_issues, "total": c_total,
            "rate": c_issues / c_total if c_total > 0 else 0,
        }
    for rank, idx in enumerate(sorted_by_quality[:50]):
        noise_analysis["top_suspicious"].append({
            "rank": rank + 1,
            "train_idx": int(idx),
            "given_label": int(train_labels[idx]),
            "model_pred": int(cv_pred[idx]),
            "quality_score": float(quality_scores[idx]),
        })

    # ── Step 4: Clean and retrain ──────────────────────────────
    clean_metrics = None
    if args.clean_method != "none":
        print("\n" + "=" * 60)
        print(f"[Step 4] Clean method: {args.clean_method}")
        print("=" * 60)

        if args.clean_method == "remove":
            clean_mask = ~label_issues_mask
            clean_input = train_input[clean_mask]
            clean_labels = train_labels[clean_mask]
            print(f"[Clean] removed {n_issues} noisy samples, "
                  f"remaining: {len(clean_labels)}")
        elif args.clean_method == "relabel":
            clean_input = train_input.copy()
            clean_labels = train_labels.copy()
            clean_labels[label_issues_mask] = cv_pred[label_issues_mask]
            n_changed = label_issues_mask.sum()
            print(f"[Clean] relabeled {n_changed} samples with model predictions")

        print(f"[Clean] label distribution after cleaning:")
        for c in range(num_classes):
            print(f"  class {c}: {(clean_labels == c).sum()}")

        # Retrain on cleaned data
        clean_model = MLPClassifier(input_dim, args.hidden_dim, num_classes, args.dropout)
        clean_t = torch.from_numpy(clean_input.astype(np.float32))
        clean_labels_t = torch.from_numpy(clean_labels).long()

        clean_loader = DataLoader(
            TensorDataset(clean_t, clean_labels_t),
            batch_size=args.batch_size, shuffle=True,
        )
        train_model(
            clean_model, clean_loader, test_t, test_labels_t,
            args, None, device, num_classes, show_progress,
        )

        clean_pred, _ = predict(clean_model, test_t, device)
        clean_metrics = evaluate(clean_pred, test_labels, num_classes)
        print(f"[Clean] accuracy={clean_metrics['accuracy']:.4f}, "
              f"macro_f1={clean_metrics['macro_f1']:.4f}, "
              f"mae={clean_metrics['mean_abs_error']:.4f}")

        # Compare
        acc_delta = clean_metrics["accuracy"] - base_metrics["accuracy"]
        f1_delta = clean_metrics["macro_f1"] - base_metrics["macro_f1"]
        print(f"\n[Compare] Baseline → Cleaned:")
        print(f"  Accuracy: {base_metrics['accuracy']:.4f} → "
              f"{clean_metrics['accuracy']:.4f} ({acc_delta:+.4f})")
        print(f"  Macro F1: {base_metrics['macro_f1']:.4f} → "
              f"{clean_metrics['macro_f1']:.4f} ({f1_delta:+.4f})")

    # ── Save results ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[Summary]")
    print("=" * 60)
    print(f"  Baseline:  acc={base_metrics['accuracy']:.4f}, f1={base_metrics['macro_f1']:.4f}")
    if clean_metrics:
        print(f"  Cleaned:   acc={clean_metrics['accuracy']:.4f}, "
              f"f1={clean_metrics['macro_f1']:.4f} ({args.clean_method})")
    print(f"  Noise rate: {issue_rate*100:.1f}% ({n_issues}/{len(train_labels)})")

    results = {
        "train_ratio": args.train_ratio,
        "train_rows": int(len(train_labels)),
        "test_rows": int(len(test_labels)),
        "num_classes": num_classes,
        "cv_folds": args.cv_folds,
        "clean_method": args.clean_method,
        "baseline_metrics": base_metrics,
        "noise_rate": float(issue_rate),
        "label_issues_count": int(n_issues),
    }
    if clean_metrics:
        results["clean_metrics"] = clean_metrics
        results["clean_train_rows"] = int(len(clean_labels))

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(to_serializable(results), f, indent=2)

    with open(output_dir / "noise_analysis.json", "w") as f:
        json.dump(to_serializable(noise_analysis), f, indent=2)

    print(f"\n[info] results saved to {output_dir}")


if __name__ == "__main__":
    main()
