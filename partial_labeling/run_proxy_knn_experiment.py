import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import numpy._core  # type: ignore
except Exception:
    import numpy.core as _numpy_core  # type: ignore

    sys.modules.setdefault("numpy._core", _numpy_core)
    sys.modules.setdefault("numpy._core.multiarray", _numpy_core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", _numpy_core.numeric)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone 10% seed-label + kNN propagation experiment."
    )
    parser.add_argument("--config", default="template.py")
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--embedded_pt", default=None)
    parser.add_argument("--output_dir", default="runs/proxy_knn_exp1")
    parser.add_argument("--feature_key", default="embed_text")
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--budget", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--knn_k", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--mode", choices=["dev_teacher", "llm"], default="dev_teacher")
    parser.add_argument("--teacher_score_key", default="bin_score")
    parser.add_argument("--output_score_key", default="bin_score_proxy")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--knn_batch_size", type=int, default=1024)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_python_config(config_path: str) -> SimpleNamespace:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    module_name = f"proxy_cfg_{cfg_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, cfg_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = {
        key: value
        for key, value in module.__dict__.items()
        if not key.startswith("_")
    }
    return SimpleNamespace(**values)


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_hf_cache(cache_root: Path) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))


def load_rows(dataset_path: str, cache_dir: str) -> list[dict[str, Any]]:
    lower = dataset_path.lower()
    if lower.endswith(".parquet"):
        dataset = load_dataset("parquet", data_files=dataset_path, cache_dir=cache_dir)["train"]
    elif lower.endswith(".jsonl") or lower.endswith(".json"):
        dataset = load_dataset("json", data_files=dataset_path, cache_dir=cache_dir)["train"]
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path}")
    return [dict(sample) for sample in dataset]


def load_embeddings_from_pt(embedded_pt: str) -> tuple[np.ndarray, np.ndarray]:
    path = Path(embedded_pt)
    if path.is_dir():
        files = sorted(path.glob("embedded_*.pt"))
    elif "*" in embedded_pt:
        files = sorted(Path(".").glob(embedded_pt))
    else:
        files = [path]

    if not files:
        raise FileNotFoundError(f"No embedding files found from: {embedded_pt}")

    all_features = []
    all_indices = []
    for file_path in files:
        obj = torch.load(file_path)
        if not hasattr(obj, "feature") or not hasattr(obj, "index"):
            raise ValueError(f"Unsupported embedded object at {file_path}")
        all_features.append(np.asarray(obj.feature))
        all_indices.append(np.asarray(obj.index))

    features = np.concatenate(all_features, axis=0)
    indices = np.concatenate(all_indices, axis=0).astype(np.int64)
    if len(indices) != len(np.unique(indices)):
        raise ValueError("Duplicated row indices found in embedded files.")
    order = np.argsort(indices)
    return features[order], indices[order]


def mean_pooling(model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


def build_text_embeddings(
    texts: list[str],
    model_name: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    outputs = []
    for start in range(0, len(texts), batch_size):
        batch_text = texts[start : start + batch_size]
        encoded = tokenizer(batch_text, padding=True, truncation=True, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            model_output = model(**encoded)
        embedding = mean_pooling(model_output, encoded["attention_mask"])
        embedding = F.normalize(embedding, p=2, dim=1)
        outputs.append(embedding.cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def resolve_num_classes(
    num_classes: int | None,
    rows: list[dict[str, Any]],
    teacher_score_key: str | None,
    cfg: Any,
) -> int:
    if num_classes is not None:
        return num_classes
    if teacher_score_key is not None and rows and teacher_score_key in rows[0]:
        labels = [int(row[teacher_score_key]) for row in rows]
        return int(max(labels) + 1)
    if hasattr(cfg, "num_classes"):
        return int(cfg.num_classes)
    raise ValueError("num_classes is required when teacher_score_key is unavailable.")


def sample_seed_indices(num_rows: int, budget: float, seed: int) -> np.ndarray:
    if budget <= 0:
        raise ValueError("budget must be > 0")
    if budget < 1:
        sample_size = max(1, int(round(num_rows * budget)))
    else:
        sample_size = int(budget)
    sample_size = min(sample_size, num_rows)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_rows, size=sample_size, replace=False))


def knn_topk(
    embeddings: np.ndarray,
    k: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    if k < 1:
        raise ValueError("knn_k must be >= 1")
    emb = torch.from_numpy(embeddings).to(device)
    emb = F.normalize(emb, dim=1)
    num_rows = emb.shape[0]
    k_eff = min(k, num_rows - 1) if num_rows > 1 else 0
    if k_eff == 0:
        return np.zeros((num_rows, 0), dtype=np.int64), np.zeros((num_rows, 0), dtype=np.float32)

    all_idx = []
    all_sim = []
    for start in range(0, num_rows, batch_size):
        end = min(start + batch_size, num_rows)
        batch = emb[start:end]
        sims = torch.matmul(batch, emb.T)
        row_ids = torch.arange(start, end, device=device)
        sims[torch.arange(end - start, device=device), row_ids] = -1e9
        top_sims, top_idx = torch.topk(sims, k=k_eff, dim=1, largest=True, sorted=True)
        all_idx.append(top_idx.cpu().numpy().astype(np.int64))
        all_sim.append(top_sims.cpu().numpy().astype(np.float32))

    return np.concatenate(all_idx, axis=0), np.concatenate(all_sim, axis=0)


def propagate_scores(
    teacher_labels: np.ndarray,
    seed_indices: np.ndarray,
    knn_indices: np.ndarray,
    knn_sims: np.ndarray,
    num_classes: int,
    tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    num_rows = len(teacher_labels)
    if tau <= 0:
        raise ValueError("tau must be > 0")

    seed_mask = np.zeros(num_rows, dtype=bool)
    seed_mask[seed_indices] = True

    priors = np.zeros(num_classes, dtype=np.float64)
    for idx in seed_indices:
        priors[teacher_labels[idx]] += 1.0
    priors /= priors.sum()

    proxy_scores = np.zeros(num_rows, dtype=np.int64)
    proxy_conf = np.zeros(num_rows, dtype=np.float32)
    proxy_source = np.array(["propagated"] * num_rows, dtype=object)
    fallback_count = 0

    for idx in range(num_rows):
        if seed_mask[idx]:
            proxy_scores[idx] = int(teacher_labels[idx])
            proxy_conf[idx] = 1.0
            proxy_source[idx] = "seed"
            continue

        neighbors = knn_indices[idx]
        sims = knn_sims[idx]
        labeled_neighbor_mask = seed_mask[neighbors]
        if not np.any(labeled_neighbor_mask):
            probs = priors.copy()
            fallback_count += 1
        else:
            nbr_idx = neighbors[labeled_neighbor_mask]
            nbr_sims = sims[labeled_neighbor_mask]
            weights = np.exp(np.clip(nbr_sims / tau, -50, 50)).astype(np.float64)
            class_votes = np.zeros(num_classes, dtype=np.float64)
            for row, weight in zip(nbr_idx, weights):
                class_votes[int(teacher_labels[row])] += float(weight)
            denom = class_votes.sum()
            probs = class_votes / denom if denom > 0 else priors.copy()

        score = int(np.argmax(probs))
        proxy_scores[idx] = score
        proxy_conf[idx] = float(probs[score])

    return proxy_scores, proxy_conf, proxy_source, fallback_count


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        mat[int(true_label), int(pred_label)] += 1
    return mat


def macro_f1_score(conf_mat: np.ndarray) -> float:
    f1_list = []
    for cls_idx in range(conf_mat.shape[0]):
        tp = conf_mat[cls_idx, cls_idx]
        fp = conf_mat[:, cls_idx].sum() - tp
        fn = conf_mat[cls_idx, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_list.append(float(f1))
    return float(np.mean(f1_list))


def confidence_bucket_stats(y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray) -> list[dict[str, Any]]:
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    stats = []
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidence >= low) & (confidence < high)
        count = int(mask.sum())
        if count == 0:
            acc = None
        else:
            acc = float((y_true[mask] == y_pred[mask]).mean())
        stats.append({"bucket": f"[{low:.1f},{min(high,1.0):.1f})", "count": count, "accuracy": acc})
    return stats


def build_metrics(
    teacher_labels: np.ndarray,
    proxy_scores: np.ndarray,
    proxy_source: np.ndarray,
    proxy_confidence: np.ndarray,
    num_classes: int,
    fallback_count: int,
) -> dict[str, Any]:
    conf = confusion_matrix(teacher_labels, proxy_scores, num_classes)
    acc = float((teacher_labels == proxy_scores).mean())
    macro_f1 = macro_f1_score(conf)
    num_seed = int(np.sum(proxy_source == "seed"))
    total = len(proxy_source)
    metrics = {
        "num_samples": total,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": conf.tolist(),
        "coverage": {
            "seed_count": num_seed,
            "seed_ratio": num_seed / total if total > 0 else 0.0,
            "propagated_count": total - num_seed,
            "propagated_ratio": (total - num_seed) / total if total > 0 else 0.0,
        },
        "fallback_prior_count": int(fallback_count),
        "confidence_buckets": confidence_bucket_stats(teacher_labels, proxy_scores, proxy_confidence),
    }
    return metrics


def merge_outputs(
    rows: list[dict[str, Any]],
    output_score_key: str,
    proxy_scores: np.ndarray,
    proxy_confidence: np.ndarray,
    proxy_source: np.ndarray,
) -> list[dict[str, Any]]:
    merged = []
    for idx, row in enumerate(rows):
        item = dict(row)
        item[output_score_key] = int(proxy_scores[idx])
        item["proxy_confidence"] = float(proxy_confidence[idx])
        item["proxy_source"] = str(proxy_source[idx])
        merged.append(item)
    return merged


def ensure_output_dir(path: str) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main() -> None:
    args = parse_args()
    if args.mode == "llm":
        raise NotImplementedError("llm mode is not implemented yet. Use --mode dev_teacher.")
    if not args.dataset_path and not args.embedded_pt:
        raise ValueError("At least one of --dataset_path or --embedded_pt must be provided.")

    cfg = load_python_config(args.config)
    device = resolve_device(args.device)
    output_dir = ensure_output_dir(args.output_dir)
    hf_cache_root = output_dir / ".hf_cache"
    configure_hf_cache(hf_cache_root)

    rows = load_rows(args.dataset_path, cache_dir=str(hf_cache_root / "datasets")) if args.dataset_path else []
    if not rows and args.mode == "dev_teacher":
        raise ValueError("dataset_path is required in dev_teacher mode.")

    if args.embedded_pt:
        embeddings, embedding_indices = load_embeddings_from_pt(args.embedded_pt)
        if rows:
            if len(rows) != len(embeddings):
                raise ValueError(
                    f"dataset rows ({len(rows)}) and embeddings ({len(embeddings)}) size mismatch."
                )
            if not np.array_equal(embedding_indices, np.arange(len(rows), dtype=np.int64)):
                rows = [rows[int(i)] for i in embedding_indices.tolist()]
    else:
        if not rows:
            raise ValueError("dataset_path is required when embedded_pt is not provided.")
        if args.feature_key not in rows[0]:
            raise KeyError(f"feature_key '{args.feature_key}' not found in dataset.")
        texts = [str(row.get(args.feature_key, "")) for row in rows]
        embeddings = build_text_embeddings(
            texts=texts,
            model_name=cfg.embedding_model,
            device=device,
            batch_size=args.batch_size,
        )

    num_rows = len(embeddings)
    if num_rows < 2:
        raise ValueError("At least 2 samples are required for kNN propagation.")

    if args.teacher_score_key not in rows[0]:
        raise KeyError(f"teacher_score_key '{args.teacher_score_key}' not found in dataset.")
    teacher_labels = np.array([int(row[args.teacher_score_key]) for row in rows], dtype=np.int64)

    num_classes = resolve_num_classes(
        num_classes=args.num_classes,
        rows=rows,
        teacher_score_key=args.teacher_score_key,
        cfg=cfg,
    )
    if teacher_labels.min() < 0 or teacher_labels.max() >= num_classes:
        raise ValueError(
            f"teacher labels must be within [0, {num_classes - 1}], got range "
            f"[{teacher_labels.min()}, {teacher_labels.max()}]."
        )

    seed_indices = sample_seed_indices(num_rows=num_rows, budget=args.budget, seed=args.seed)
    knn_indices, knn_sims = knn_topk(
        embeddings=embeddings,
        k=args.knn_k,
        batch_size=args.knn_batch_size,
        device=device,
    )
    proxy_scores, proxy_confidence, proxy_source, fallback_count = propagate_scores(
        teacher_labels=teacher_labels,
        seed_indices=seed_indices,
        knn_indices=knn_indices,
        knn_sims=knn_sims,
        num_classes=num_classes,
        tau=args.tau,
    )

    merged_rows = merge_outputs(
        rows=rows,
        output_score_key=args.output_score_key,
        proxy_scores=proxy_scores,
        proxy_confidence=proxy_confidence,
        proxy_source=proxy_source,
    )

    metrics = build_metrics(
        teacher_labels=teacher_labels,
        proxy_scores=proxy_scores,
        proxy_source=proxy_source,
        proxy_confidence=proxy_confidence,
        num_classes=num_classes,
        fallback_count=fallback_count,
    )

    meta = {
        "config": args.config,
        "dataset_path": args.dataset_path,
        "embedded_pt": args.embedded_pt,
        "feature_key": args.feature_key,
        "teacher_score_key": args.teacher_score_key,
        "output_score_key": args.output_score_key,
        "num_classes": int(num_classes),
        "num_samples": int(num_rows),
        "mode": args.mode,
        "budget": float(args.budget),
        "seed": int(args.seed),
        "seed_count": int(len(seed_indices)),
        "knn_k": int(args.knn_k),
        "tau": float(args.tau),
        "embedding_model": str(cfg.embedding_model),
        "embedding_shape": [int(num_rows), int(embeddings.shape[1])],
        "device": str(device),
    }

    proxy_dataset_path = output_dir / "proxy_dataset.json"
    metrics_path = output_dir / "metrics.json"
    meta_path = output_dir / "proxy_meta.json"

    with proxy_dataset_path.open("w", encoding="utf-8") as file:
        json.dump(merged_rows, file, ensure_ascii=False, indent=2)
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)

    print(f"[proxy] output_dir={output_dir}")
    print(f"[proxy] dataset={proxy_dataset_path}")
    print(f"[proxy] metrics={metrics_path}")
    print(f"[proxy] meta={meta_path}")
    print(
        "[proxy] summary: "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}, "
        f"seed_ratio={metrics['coverage']['seed_ratio']:.4f}"
    )


if __name__ == "__main__":
    main()
