import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Proxy kNN evaluation on a single json/jsonl/parquet dataset with precomputed embeddings. "
            "Split by train_ratio, use train labels as source, evaluate on test labels."
        )
    )
    parser.add_argument("--dataset_path", default="raw_data/tulu_300k_with_embeddings.parquet")
    parser.add_argument("--embedding_key", default="embeddings")
    parser.add_argument("--embedding_path", default=None,
                        help="Path to external .npy embedding file (overrides --embedding_key column)")
    parser.add_argument("--label_key", default="gpt_scores")
    parser.add_argument("--prediction_key", default="proxy_knn_label")
    parser.add_argument("--train_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--knn_k", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--predict_mode", choices=["vote", "regression"], default="vote",
                        help="vote: softmax class voting; regression: weighted mean of neighbor labels")
    parser.add_argument("--weight_type", choices=["softmax", "power"], default="softmax",
                        help="softmax: exp(sim/tau); power: sim^alpha (distance-aware)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Exponent for power weighting (higher = more weight on closer neighbors)")
    parser.add_argument("--sim_threshold", type=float, default=0.0,
                        help="Ignore neighbors with similarity below this threshold")
    parser.add_argument("--prior_calibration", action="store_true",
                        help="Divide class votes by training class prior to counter imbalance")
    parser.add_argument("--knn_batch_size", type=int, default=1024)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--data_pool_size",
        type=int,
        default=None,
        help="Limit number of rows loaded into the experiment data pool.",
    )
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--output_dir", default="runs/proxy_knn_single_dataset")
    parser.add_argument("--disable_tqdm", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [to_serializable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    return value


def stack_embedding_column(
    df: pd.DataFrame,
    embedding_key: str,
    show_progress: bool = True,
) -> np.ndarray:
    if embedding_key not in df.columns:
        raise KeyError(f"embedding_key '{embedding_key}' not found in dataset")

    vectors = []
    dim = None
    for i, item in enumerate(
        tqdm(
            df[embedding_key].tolist(),
            desc="stack embeddings",
            unit="row",
            disable=not show_progress,
        )
    ):
        arr = np.asarray(item, dtype=np.float32).reshape(-1)
        if dim is None:
            dim = arr.shape[0]
        elif arr.shape[0] != dim:
            raise ValueError(
                f"Inconsistent embedding dim at row {i}: got {arr.shape[0]}, expected {dim}"
            )
        vectors.append(arr)
    return np.stack(vectors, axis=0).astype(np.float32)


def load_dataframe(dataset_path: str) -> pd.DataFrame:
    lower = dataset_path.lower()
    if lower.endswith(".parquet"):
        return pd.read_parquet(dataset_path)
    if lower.endswith(".jsonl"):
        return pd.read_json(dataset_path, lines=True)
    if lower.endswith(".json"):
        try:
            return pd.read_json(dataset_path, lines=True)
        except ValueError:
            return pd.read_json(dataset_path)
    raise ValueError(f"Unsupported dataset format: {dataset_path}. Use json/jsonl/parquet.")


def resolve_num_classes(
    user_num_classes: int | None,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
) -> int:
    if user_num_classes is not None:
        return user_num_classes
    return int(max(train_labels.max(), test_labels.max()) + 1)


def split_indices(num_rows: int, train_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")
    train_count = int(round(num_rows * train_ratio))
    train_count = max(1, min(train_count, num_rows - 1))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_rows)
    train_idx = np.sort(perm[:train_count])
    test_idx = np.sort(perm[train_count:])
    return train_idx, test_idx


def knn_query_train(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    knn_k: int,
    batch_size: int,
    device: torch.device,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if knn_k < 1:
        raise ValueError("knn_k must be >= 1")

    train = torch.from_numpy(train_embeddings).to(device)
    test = torch.from_numpy(test_embeddings).to(device)
    train = F.normalize(train, dim=1)
    test = F.normalize(test, dim=1)

    k_eff = min(knn_k, train.shape[0])
    all_idx = []
    all_sim = []
    for start in tqdm(
        range(0, test.shape[0], batch_size),
        desc="knn query",
        unit="batch",
        disable=not show_progress,
    ):
        end = min(start + batch_size, test.shape[0])
        sims = torch.matmul(test[start:end], train.T)
        top_sim, top_idx = torch.topk(sims, k=k_eff, dim=1, largest=True, sorted=True)
        all_idx.append(top_idx.cpu().numpy().astype(np.int64))
        all_sim.append(top_sim.cpu().numpy().astype(np.float32))

    return np.concatenate(all_idx, axis=0), np.concatenate(all_sim, axis=0)


def compute_weights(
    neighbor_sims: np.ndarray,
    weight_type: str = "softmax",
    tau: float = 0.1,
    alpha: float = 1.0,
    sim_threshold: float = 0.0,
) -> np.ndarray:
    """Compute neighbor weights from similarities.

    weight_type:
        softmax: exp(sim / tau)
        power:   max(sim, 0) ^ alpha  (distance-aware power decay)
    sim_threshold: zero out neighbors below this similarity.
    """
    if weight_type == "power":
        # Clamp negatives to 0, then raise to power alpha
        clamped = np.maximum(neighbor_sims, 0.0).astype(np.float64)
        weights = np.power(clamped, alpha)
    else:
        weights = np.exp(np.clip(neighbor_sims / tau, -50, 50)).astype(np.float64)

    # Apply similarity threshold mask
    if sim_threshold > 0:
        mask = neighbor_sims >= sim_threshold
        weights = weights * mask.astype(np.float64)
        # Ensure at least the closest neighbor has weight (fallback)
        no_valid = weights.sum(axis=1) == 0
        if no_valid.any():
            weights[no_valid, 0] = 1.0

    return weights


def predict_proxy_scores(
    train_labels: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_sims: np.ndarray,
    num_classes: int,
    tau: float,
    predict_mode: str = "vote",
    weight_type: str = "softmax",
    alpha: float = 1.0,
    sim_threshold: float = 0.0,
    class_prior: np.ndarray | None = None,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    # Vectorized: gather neighbor labels and compute weights
    nbr_labels = train_labels[neighbor_indices]  # (N, K)
    weights = compute_weights(neighbor_sims, weight_type, tau, alpha, sim_threshold)

    if predict_mode == "regression":
        # Weighted mean of neighbor labels → round to nearest integer → clip
        w_sum = weights.sum(axis=1, keepdims=True)  # (N, 1)
        weighted_mean = (weights * nbr_labels).sum(axis=1) / w_sum.squeeze()  # (N,)
        pred = np.clip(np.round(weighted_mean), 0, num_classes - 1).astype(np.int64)
        # Confidence: 1 - normalized std of neighbor labels
        weighted_var = (weights * (nbr_labels - weighted_mean[:, None]) ** 2).sum(axis=1) / w_sum.squeeze()
        conf = np.clip(1.0 - np.sqrt(weighted_var) / (num_classes - 1), 0, 1).astype(np.float32)
    else:
        # Classification: softmax weighted voting (vectorized)
        one_hot = np.eye(num_classes, dtype=np.float64)[nbr_labels]  # (N, K, C)
        votes = (weights[:, :, None] * one_hot).sum(axis=1)  # (N, C)

        # Prior calibration: divide votes by class frequency to counter imbalance
        if class_prior is not None:
            votes = votes / class_prior[None, :]

        probs = votes / votes.sum(axis=1, keepdims=True)  # (N, C)
        pred = np.argmax(probs, axis=1).astype(np.int64)
        conf = np.take_along_axis(probs, pred[:, None], axis=1).squeeze().astype(np.float32)

    return pred, conf


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        mat[int(t), int(p)] += 1
    return mat


def macro_f1(conf_mat: np.ndarray) -> float:
    f1_values = []
    for c in range(conf_mat.shape[0]):
        tp = conf_mat[c, c]
        fp = conf_mat[:, c].sum() - tp
        fn = conf_mat[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_values.append(float(f1))
    return float(np.mean(f1_values))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(args.dataset_path)
    pool_size = args.data_pool_size if args.data_pool_size is not None else args.max_rows
    if pool_size is not None:
        if pool_size < 2:
            raise ValueError("data_pool_size must be >= 2")
        df = df.iloc[:pool_size].copy()
    df = df.reset_index(drop=True)

    if args.label_key not in df.columns:
        raise KeyError(f"label_key '{args.label_key}' not found in dataset")

    show_progress = not args.disable_tqdm

    if args.embedding_path is not None:
        print(f"[info] loading external embeddings from {args.embedding_path}")
        embeddings = np.load(args.embedding_path).astype(np.float32)
        if embeddings.shape[0] != len(df):
            raise ValueError(
                f"Embedding row count ({embeddings.shape[0]}) != dataset row count ({len(df)}). "
                f"Make sure --data_pool_size matches."
            )
        print(f"[info] external embeddings shape: {embeddings.shape}")
    else:
        embeddings = stack_embedding_column(
            df=df,
            embedding_key=args.embedding_key,
            show_progress=show_progress,
        )
    labels = np.array([int(v) for v in df[args.label_key].tolist()], dtype=np.int64)

    train_idx, test_idx = split_indices(
        num_rows=len(df),
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    train_embeddings = embeddings[train_idx]
    test_embeddings = embeddings[test_idx]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    num_classes = resolve_num_classes(
        user_num_classes=args.num_classes,
        train_labels=train_labels,
        test_labels=test_labels,
    )
    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError(
            f"Label out of range. label min/max={labels.min()}/{labels.max()}, num_classes={num_classes}"
        )

    device = resolve_device(args.device)
    neighbor_idx, neighbor_sim = knn_query_train(
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        knn_k=args.knn_k,
        batch_size=args.knn_batch_size,
        device=device,
        show_progress=show_progress,
    )
    # Compute class prior from training labels for calibration
    class_prior = None
    if args.prior_calibration:
        class_prior = np.bincount(train_labels, minlength=num_classes).astype(np.float64)
        class_prior = class_prior / class_prior.sum()
        # Avoid division by zero for classes not in training set
        class_prior = np.maximum(class_prior, 1e-10)

    proxy_pred, proxy_conf = predict_proxy_scores(
        train_labels=train_labels,
        neighbor_indices=neighbor_idx,
        neighbor_sims=neighbor_sim,
        num_classes=num_classes,
        tau=args.tau,
        predict_mode=args.predict_mode,
        weight_type=args.weight_type,
        alpha=args.alpha,
        sim_threshold=args.sim_threshold,
        class_prior=class_prior,
        show_progress=show_progress,
    )

    conf_mat = confusion_matrix(test_labels, proxy_pred, num_classes)
    metrics = {
        "dataset_path": args.dataset_path,
        "embedding_key": args.embedding_key,
        "embedding_path": args.embedding_path,
        "label_key": args.label_key,
        "num_rows": int(len(df)),
        "data_pool_size": int(pool_size) if pool_size is not None else None,
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_ratio": float(args.train_ratio),
        "seed": int(args.seed),
        "num_classes": int(num_classes),
        "knn_k": int(args.knn_k),
        "tau": float(args.tau),
        "predict_mode": args.predict_mode,
        "weight_type": args.weight_type,
        "alpha": float(args.alpha),
        "sim_threshold": float(args.sim_threshold),
        "prior_calibration": bool(args.prior_calibration),
        "device": str(device),
        "accuracy": float((proxy_pred == test_labels).mean()),
        "macro_f1": macro_f1(conf_mat),
        "mean_abs_error": float(np.abs(proxy_pred - test_labels).mean()),
        "confusion_matrix": conf_mat.tolist(),
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    predictions_path = output_dir / "test_with_proxy.jsonl"
    with predictions_path.open("w", encoding="utf-8") as file:
        for local_i, global_i in enumerate(
            tqdm(
                test_idx.tolist(),
                desc="write predictions",
                unit="row",
                disable=not show_progress,
            )
        ):
            row = df.iloc[global_i].to_dict()
            row.pop(args.embedding_key, None)  # exclude embedding to reduce file size
            row = {k: to_serializable(v) for k, v in row.items()}
            row[args.prediction_key] = int(proxy_pred[local_i])
            row["proxy_confidence"] = float(proxy_conf[local_i])
            row["ground_truth_label"] = int(test_labels[local_i])
            row["split"] = "test"
            row["row_index"] = int(global_i)
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    split_index_path = output_dir / "split_indices.json"
    with split_index_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "train_indices": train_idx.tolist(),
                "test_indices": test_idx.tolist(),
            },
            file,
            ensure_ascii=False,
        )

    print(f"[done] metrics: {metrics_path}")
    print(f"[done] predictions: {predictions_path}")
    print(f"[done] split indices: {split_index_path}")
    print(
        "[summary] "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}, "
        f"mae={metrics['mean_abs_error']:.4f}"
    )


if __name__ == "__main__":
    main()
