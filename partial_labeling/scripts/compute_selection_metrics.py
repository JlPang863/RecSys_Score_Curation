"""
Compute high-score selection metrics for proxy label generation.

For the downstream task of selecting high-quality samples (score >= 4),
computes metrics that directly measure selection quality:

1. Binary AUC (high vs low): P(score>=4) as positive class
2. Per-class AUC for class 4 and class 5
3. Precision@k: among top-k% selected by P(>=4), fraction truly >=4
4. Recall@k: among all true >=4 samples, fraction captured in top-k%

Usage:
    python compute_selection_metrics.py
    python compute_selection_metrics.py --run_dirs runs/skywork_fusion_mlp_300k_tr095
    python compute_selection_metrics.py --threshold 3  # select score >= 3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "partial_labeling"))

# Reuse infrastructure from compute_auc_all
from compute_auc_all import (
    _get_df,
    _get_emb,
    _get_text_feats,
    get_probs,
)
from partial_labeling.supervised_proxy import (
    LinearProbe,
    MLPClassifier,
    OrdinalModel,
    ResNetMLP,
)
from partial_labeling.proxy_label_generation import split_indices


def compute_selection_metrics(
    probs: np.ndarray,
    test_labels: np.ndarray,
    threshold: int = 4,
    top_k_percents: list[float] = None,
) -> dict:
    """
    Compute selection metrics for high-score filtering.

    Args:
        probs: (N, num_classes) probability matrix
        test_labels: (N,) ground truth labels
        threshold: score >= threshold is "high quality" (default: 4)
        top_k_percents: list of percentages for Precision/Recall@k
    """
    if top_k_percents is None:
        top_k_percents = [5, 10, 15, 20, 25, 30, 50]

    num_classes = probs.shape[1]
    N = len(test_labels)

    # Binary: high quality (>= threshold) vs low quality (< threshold)
    binary_true = (test_labels >= threshold).astype(int)
    high_score_prob = probs[:, threshold:].sum(axis=1)  # P(score >= threshold)
    n_positive = binary_true.sum()
    positive_rate = n_positive / N

    # Binary AUC
    try:
        binary_auc = float(roc_auc_score(binary_true, high_score_prob))
    except ValueError:
        binary_auc = None

    # Average Precision (area under PR curve)
    try:
        avg_precision = float(average_precision_score(binary_true, high_score_prob))
    except ValueError:
        avg_precision = None

    # Per-class AUC for high classes
    per_class_auc = {}
    for c in range(threshold, num_classes):
        binary = (test_labels == c).astype(int)
        if binary.sum() == 0 or binary.sum() == len(binary):
            per_class_auc[c] = None
        else:
            try:
                per_class_auc[c] = float(roc_auc_score(binary, probs[:, c]))
            except ValueError:
                per_class_auc[c] = None

    # Precision@k and Recall@k
    # Sort by P(>= threshold) descending
    sorted_indices = np.argsort(-high_score_prob)
    sorted_true = binary_true[sorted_indices]

    precision_at_k = {}
    recall_at_k = {}
    for pct in top_k_percents:
        k = max(1, int(N * pct / 100))
        selected = sorted_true[:k]
        prec = float(selected.sum() / k)
        rec = float(selected.sum() / n_positive) if n_positive > 0 else 0.0
        precision_at_k[pct] = prec
        recall_at_k[pct] = rec

    # Also compute: if we select by hard label (predicted score >= threshold)
    predicted_labels = probs.argmax(axis=1)
    predicted_high = (predicted_labels >= threshold)
    n_predicted_high = predicted_high.sum()

    if n_predicted_high > 0:
        hard_precision = float(binary_true[predicted_high].sum() / n_predicted_high)
        hard_recall = float(binary_true[predicted_high].sum() / n_positive) if n_positive > 0 else 0.0
    else:
        hard_precision = 0.0
        hard_recall = 0.0

    return {
        "threshold": threshold,
        "n_test": N,
        "n_true_high": int(n_positive),
        "true_high_rate": float(positive_rate),
        "binary_auc": binary_auc,
        "avg_precision": avg_precision,
        "per_class_auc": {str(k): v for k, v in per_class_auc.items()},
        "precision_at_k": {str(k): v for k, v in precision_at_k.items()},
        "recall_at_k": {str(k): v for k, v in recall_at_k.items()},
        # Hard label selection (argmax >= threshold)
        "hard_n_selected": int(n_predicted_high),
        "hard_select_rate": float(n_predicted_high / N),
        "hard_precision": hard_precision,
        "hard_recall": hard_recall,
    }


def process_run(run_dir: Path, device: torch.device, threshold: int) -> dict | None:
    metrics_path = run_dir / "metrics.json"
    model_path = run_dir / "model.pt"

    if not metrics_path.exists() or not model_path.exists():
        print(f"  SKIP {run_dir.name}: missing files")
        return None

    with open(metrics_path) as f:
        cfg = json.load(f)

    if "dataset_path" not in cfg:
        print(f"  SKIP {run_dir.name}: non-standard format")
        return None

    dataset_path = cfg["dataset_path"]
    label_key = cfg["label_key"]
    train_ratio = cfg["train_ratio"]
    seed = cfg.get("seed", 3)
    num_classes = cfg["num_classes"]
    data_pool_size = cfg.get("data_pool_size")
    model_type = cfg["model"]
    hidden_dim = cfg["hidden_dim"]
    num_layers = cfg.get("num_layers")
    dropout = cfg.get("dropout", 0.1)

    # Embedding source
    if "balanced_18k" in dataset_path:
        embedding_npy = "raw_data/embedding_cache/Skywork_balanced_18k_embeddings.npy"
    elif "balanced_100k" in dataset_path:
        embedding_npy = "raw_data/embedding_cache/Skywork_balanced_100k_embeddings.npy"
    else:
        embedding_npy = "raw_data/embedding_cache/Skywork_Skywork-Reward-Llama-3.1-8B_300932_embeddings.npy"

    # Load data
    df = _get_df(dataset_path, data_pool_size)
    labels = np.array([int(v) for v in df[label_key].tolist()], dtype=np.int64)
    train_idx, test_idx = split_indices(len(df), train_ratio, seed)
    test_labels = labels[test_idx]

    emb = _get_emb(embedding_npy, data_pool_size)

    # Text features
    run_name = run_dir.name
    use_text_features = any(k in run_name for k in
                            ["fusion", "classweight", "focal", "oversample", "ordinal", "balanced"])

    if use_text_features:
        from partial_labeling.feature_extraction import normalize_features
        text_feats, _ = _get_text_feats(dataset_path, data_pool_size)
        train_tf, test_tf = normalize_features(text_feats[train_idx], text_feats[test_idx])
        test_input = np.concatenate([emb[test_idx], test_tf], axis=1)
    else:
        test_input = emb[test_idx]

    input_dim = test_input.shape[1]
    is_ordinal = model_type == "ordinal"

    # Build model
    if model_type == "linear":
        model = LinearProbe(input_dim, num_classes, dropout=dropout)
    elif model_type == "mlp":
        model = MLPClassifier(input_dim, hidden_dim, num_classes, dropout=dropout)
    elif model_type == "resnet":
        model = ResNetMLP(input_dim, hidden_dim, num_classes,
                          num_layers=num_layers or 4, dropout=dropout)
    elif model_type == "ordinal":
        model = OrdinalModel(input_dim, hidden_dim, num_classes,
                             num_layers=num_layers or 4, dropout=dropout)
    else:
        print(f"  SKIP {run_dir.name}: unknown model '{model_type}'")
        return None

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    test_emb_t = torch.from_numpy(test_input.astype(np.float32))
    probs = get_probs(model, test_emb_t, device, is_ordinal=is_ordinal)

    # Compute selection metrics
    metrics = compute_selection_metrics(probs, test_labels, threshold=threshold)

    print(f"  {run_dir.name}: BinAUC={metrics['binary_auc']:.4f}, "
          f"AP={metrics['avg_precision']:.4f}, "
          f"P@20={metrics['precision_at_k']['20']:.4f}, "
          f"R@20={metrics['recall_at_k']['20']:.4f}, "
          f"HardP={metrics['hard_precision']:.4f}, "
          f"HardR={metrics['hard_recall']:.4f}")

    return {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "accuracy": cfg.get("accuracy"),
        "macro_f1": cfg.get("macro_f1"),
        "auc_macro": cfg.get("auc_macro"),
        "train_ratio": train_ratio,
        "data_pool_size": data_pool_size,
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dirs", nargs="*", default=None)
    parser.add_argument("--output", default="results/selection_metrics.json")
    parser.add_argument("--threshold", type=int, default=4,
                        help="Score >= threshold is 'high quality' (default: 4)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    print(f"High quality threshold: score >= {args.threshold}")

    if args.run_dirs:
        run_dirs = [Path(d) for d in args.run_dirs]
    else:
        run_dirs = sorted(Path("runs").glob("skywork_*"))

    print(f"Found {len(run_dirs)} directories\n")

    all_results = []
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        result = process_run(run_dir, device, args.threshold)
        if result:
            all_results.append(result)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Summary table
    print(f"\n{'='*130}")
    print(f"{'Run':<50} {'Acc':>6} {'F1':>6} {'mAUC':>6} "
          f"{'BinAUC':>7} {'AP':>6} "
          f"{'P@10':>6} {'P@20':>6} {'R@20':>6} "
          f"{'HardP':>6} {'HardR':>6}")
    print(f"{'-'*130}")
    for r in sorted(all_results, key=lambda x: x.get("binary_auc") or 0, reverse=True):
        ba = f"{r['binary_auc']:.4f}" if r['binary_auc'] else "N/A"
        ap = f"{r['avg_precision']:.4f}" if r['avg_precision'] else "N/A"
        print(f"{r['run_name']:<50} "
              f"{r.get('accuracy', 0):>6.4f} {r.get('macro_f1', 0):>6.4f} "
              f"{r.get('auc_macro', 0) or 0:>6.4f} "
              f"{ba:>7} {ap:>6} "
              f"{r['precision_at_k']['10']:>6.4f} {r['precision_at_k']['20']:>6.4f} "
              f"{r['recall_at_k']['20']:>6.4f} "
              f"{r['hard_precision']:>6.4f} {r['hard_recall']:>6.4f}")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
