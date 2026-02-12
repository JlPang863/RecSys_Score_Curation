"""
Hyperparameter search for proxy kNN.
Searches over K, TAU, alpha, weight_type, predict_mode, sim_threshold
on a fixed train/test split, then prints a sorted leaderboard.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from proxy_label_generation import (
    confusion_matrix,
    knn_query_train,
    load_dataframe,
    macro_f1,
    predict_proxy_scores,
    resolve_device,
    resolve_num_classes,
    split_indices,
    stack_embedding_column,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter search for proxy kNN")
    parser.add_argument("--dataset_path", default="raw_data/tulu_300k_with_embeddings.parquet")
    parser.add_argument("--embedding_key", default="embeddings")
    parser.add_argument("--label_key", default="gpt_scores")
    parser.add_argument("--train_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--knn_batch_size", type=int, default=1024)
    parser.add_argument("--data_pool_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default="runs/hyperparam_search")
    parser.add_argument("--max_k", type=int, default=200,
                        help="Max K to search (knn_query will fetch this many neighbors)")
    return parser.parse_args()


def evaluate(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_sims: np.ndarray,
    k: int,
    tau: float,
    predict_mode: str,
    weight_type: str,
    alpha: float,
    sim_threshold: float,
    num_classes: int,
    class_prior: np.ndarray | None = None,
) -> dict:
    """Run prediction with given hyperparams and return metrics."""
    idx_k = neighbor_indices[:, :k]
    sim_k = neighbor_sims[:, :k]

    pred, conf = predict_proxy_scores(
        train_labels=train_labels,
        neighbor_indices=idx_k,
        neighbor_sims=sim_k,
        num_classes=num_classes,
        tau=tau,
        predict_mode=predict_mode,
        weight_type=weight_type,
        alpha=alpha,
        sim_threshold=sim_threshold,
        class_prior=class_prior,
        show_progress=False,
    )
    conf_mat = confusion_matrix(test_labels, pred, num_classes)
    return {
        "k": k,
        "tau": tau,
        "predict_mode": predict_mode,
        "weight_type": weight_type,
        "alpha": alpha,
        "sim_threshold": sim_threshold,
        "prior_calibration": class_prior is not None,
        "accuracy": float((pred == test_labels).mean()),
        "macro_f1": macro_f1(conf_mat),
        "mae": float(np.abs(pred - test_labels).mean()),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = load_dataframe(args.dataset_path)
    if args.data_pool_size is not None:
        df = df.iloc[: args.data_pool_size].copy()
    df = df.reset_index(drop=True)

    embeddings = stack_embedding_column(df, args.embedding_key)
    labels = np.array([int(v) for v in df[args.label_key].tolist()], dtype=np.int64)

    train_idx, test_idx = split_indices(len(df), args.train_ratio, args.seed)
    train_emb = embeddings[train_idx]
    test_emb = embeddings[test_idx]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    num_classes = resolve_num_classes(args.num_classes, train_labels, test_labels)
    device = resolve_device(args.device)

    print(f"Querying {args.max_k} nearest neighbors...")
    neighbor_idx, neighbor_sim = knn_query_train(
        train_embeddings=train_emb,
        test_embeddings=test_emb,
        knn_k=args.max_k,
        batch_size=args.knn_batch_size,
        device=device,
    )

    # ========================
    # Search grid
    # ========================
    k_values = [5, 10, 20, 50, 100, 200]
    sim_thresholds = [0.0, 0.3, 0.5, 0.7]
    modes = ["vote", "regression"]

    # Group 1: softmax weighting (tau matters, alpha irrelevant)
    tau_values = [0.01, 0.05, 0.1, 0.5, 1.0]
    # Group 2: power weighting (alpha matters, tau irrelevant)
    alpha_values = [1, 2, 4, 8, 16]

    # Compute class prior for calibration experiments
    class_prior = np.bincount(train_labels, minlength=num_classes).astype(np.float64)
    class_prior = class_prior / class_prior.sum()
    class_prior = np.maximum(class_prior, 1e-10)

    prior_options = [None, class_prior]  # None = no calibration, class_prior = with calibration

    combos = []
    for k in k_values:
        for mode in modes:
            for st in sim_thresholds:
                for prior in prior_options:
                    # softmax configs
                    for tau in tau_values:
                        combos.append((k, tau, mode, "softmax", 1.0, st, prior))
                    # power configs (only without prior, since prior only applies to vote)
                    if mode == "vote":
                        for alpha in alpha_values:
                            combos.append((k, 0.1, mode, "power", alpha, st, prior))
                    else:
                        for alpha in alpha_values:
                            combos.append((k, 0.1, mode, "power", alpha, st, None))

    print(f"Searching {len(combos)} combinations...")

    results = []
    for k, tau, mode, wt, alpha, st, prior in tqdm(combos, desc="hyperparam search"):
        if k > args.max_k:
            continue
        r = evaluate(train_labels, test_labels, neighbor_idx, neighbor_sim,
                     k, tau, mode, wt, alpha, st, num_classes, class_prior=prior)
        results.append(r)

    # Sort by accuracy descending
    results.sort(key=lambda x: x["accuracy"], reverse=True)

    # Print leaderboard
    header = (f"{'Rank':>4} | {'Mode':>10} | {'Weight':>7} | {'K':>5} | {'TAU':>6} | "
              f"{'Alpha':>5} | {'SimThr':>6} | {'Prior':>5} | {'Accuracy':>10} | {'Macro F1':>10} | {'MAE':>8}")
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for i, r in enumerate(results[:40]):
        tau_str = f"{r['tau']:.2f}" if r["weight_type"] == "softmax" else "  -  "
        alpha_str = f"{r['alpha']:>5.0f}" if r["weight_type"] == "power" else "  -  "
        prior_str = "  Y  " if r.get("prior_calibration") else "  N  "
        print(
            f"{i+1:>4} | {r['predict_mode']:>10} | {r['weight_type']:>7} | {r['k']:>5} | "
            f"{tau_str:>6} | {alpha_str:>5} | {r['sim_threshold']:>6.2f} | {prior_str:>5} | "
            f"{r['accuracy']:>10.4f} | {r['macro_f1']:>10.4f} | {r['mae']:>8.4f}"
        )
    print("=" * len(header))

    # Save full results
    results_path = output_dir / "search_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    best = results[0]
    print(f"\nBest: weight_type={best['weight_type']}, K={best['k']}, "
          f"tau={best['tau']}, alpha={best['alpha']}, mode={best['predict_mode']}, "
          f"sim_threshold={best['sim_threshold']}, prior_calibration={best.get('prior_calibration', False)}")
    print(f"  accuracy={best['accuracy']:.4f}, macro_f1={best['macro_f1']:.4f}, mae={best['mae']:.4f}")
    print(f"\nFull results ({len(results)} configs) saved to: {results_path}")


if __name__ == "__main__":
    main()
