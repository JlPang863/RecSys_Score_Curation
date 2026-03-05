"""
Apply ScoreCurationPipeline (denoise) to student model proxy labels,
using pre-computed Skywork embeddings instead of re-encoding with BGE.

Usage:
    python run_denoise_proxy_labels.py \
        --proxy_jsonl runs/skywork_fusion_mlp_150k_tr030/test_with_proxy.jsonl \
        --embedding_npy raw_data/embedding_cache/Skywork_Skywork-Reward-Llama-3.1-8B_300932_embeddings.npy \
        --output_dir results/denoise_proxy_150k_tr030 \
        --confidence_prob 0.5
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import Counter

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from docta.utils.config import Config
from docta.datasets.customize import CustomizedDataset
from docta.apis import DetectLabel, DetectFeature
from docta.core.report import Report
from datasets import load_dataset as hf_load_dataset

from score_curation.data_curation import score_curating


def evaluate(labels_pred, labels_true, num_classes, tag=""):
    """Compute accuracy, macro F1, MAE, and per-class accuracy."""
    labels_pred = np.array(labels_pred)
    labels_true = np.array(labels_true)
    n = len(labels_true)

    acc = (labels_pred == labels_true).mean()
    mae = np.abs(labels_pred - labels_true).mean()

    f1s = []
    per_class = {}
    for c in range(num_classes):
        tp = ((labels_pred == c) & (labels_true == c)).sum()
        fp = ((labels_pred == c) & (labels_true != c)).sum()
        fn = ((labels_pred != c) & (labels_true == c)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1s.append(f1)
        count = int((labels_true == c).sum())
        class_acc = float((labels_pred[labels_true == c] == c).mean()) if count > 0 else 0.0
        per_class[c] = {"acc": class_acc, "count": count}

    macro_f1 = float(np.mean(f1s))

    if tag:
        print(f"\n[{tag}]")
    print(f"  accuracy={acc:.4f}, macro_f1={macro_f1:.4f}, MAE={mae:.4f}, n={n}")
    for c in range(num_classes):
        info = per_class[c]
        print(f"  class {c}: acc={info['acc']:.4f}  (n={info['count']})")

    return {
        "accuracy": float(acc),
        "macro_f1": macro_f1,
        "mean_abs_error": float(mae),
        **{f"class_{c}_acc": per_class[c]["acc"] for c in range(num_classes)},
        **{f"class_{c}_count": per_class[c]["count"] for c in range(num_classes)},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Denoise proxy labels using pre-computed embeddings"
    )
    parser.add_argument("--proxy_jsonl", required=True,
                        help="Path to test_with_proxy.jsonl from student model")
    parser.add_argument("--embedding_npy", required=True,
                        help="Path to pre-computed embeddings .npy (full dataset)")
    parser.add_argument("--output_dir", default="results/denoise_proxy",
                        help="Output directory for denoised results")
    parser.add_argument("--config_path", default="template.py",
                        help="Path to docta config (for hoc_cfg, detect_cfg)")
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--confidence_prob", type=float, default=0.5,
                        help="Confidence threshold for score revision")
    parser.add_argument("--score_key", default="proxy_supervised_label",
                        help="Score field to denoise")
    parser.add_argument("--ground_truth_key", default="gpt_scores",
                        help="Ground truth label field for evaluation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset_name = os.path.basename(os.path.dirname(args.proxy_jsonl))

    print("=" * 60)
    print("Denoise Proxy Labels (pre-computed embeddings)")
    print("=" * 60)
    print(f"  proxy_jsonl:    {args.proxy_jsonl}")
    print(f"  embedding_npy:  {args.embedding_npy}")
    print(f"  output_dir:     {args.output_dir}")
    print(f"  score_key:      {args.score_key}")
    print(f"  ground_truth:   {args.ground_truth_key}")
    print(f"  num_classes:    {args.num_classes}")
    print(f"  confidence:     {args.confidence_prob}")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Step 1: Load proxy labels + ground truth
    # ----------------------------------------------------------------
    print("\n[Step 1] Loading proxy labels...")
    with open(args.proxy_jsonl, "r") as f:
        records = [json.loads(line) for line in f]

    proxy_labels = np.array([r[args.score_key] for r in records])
    gt_labels = np.array([r[args.ground_truth_key] for r in records])
    row_indices = np.array([r["row_index"] for r in records])
    n_samples = len(records)

    print(f"  Loaded {n_samples} samples")
    print(f"  Proxy label dist: {dict(sorted(Counter(proxy_labels.tolist()).items()))}")
    print(f"  GT label dist:    {dict(sorted(Counter(gt_labels.tolist()).items()))}")

    baseline_metrics = evaluate(proxy_labels, gt_labels, args.num_classes,
                                tag="BEFORE denoise")

    # ----------------------------------------------------------------
    # Step 2: Load pre-computed embeddings and extract test rows
    # ----------------------------------------------------------------
    print("\n[Step 2] Loading pre-computed embeddings...")
    all_embeddings = np.load(args.embedding_npy)
    print(f"  Full embedding shape: {all_embeddings.shape}")

    embeddings = all_embeddings[row_indices]
    print(f"  Selected test embeddings: {embeddings.shape}")

    # ----------------------------------------------------------------
    # Step 3: Build duplicated CustomizedDataset (required by docta)
    # The pipeline internally duplicates data for its detection algorithm.
    # ----------------------------------------------------------------
    print("\n[Step 3] Building dataset for detection...")
    features_dup = np.concatenate([embeddings, embeddings], axis=0)
    labels_dup = np.concatenate([proxy_labels, proxy_labels], axis=0)
    index_dup = np.concatenate([np.arange(n_samples), np.arange(n_samples)], axis=0)

    dataset = CustomizedDataset(
        feature=features_dup,
        label=labels_dup,
        index=index_dup,
    )
    print(f"  Duplicated dataset size: {len(dataset)} (2 x {n_samples})")

    # ----------------------------------------------------------------
    # Step 4: Load docta config and run detection
    # ----------------------------------------------------------------
    print("\n[Step 4] Running label error detection (SimiFeat)...")
    cfg = Config.fromfile(args.config_path)
    cfg.dataset_type = dataset_name
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.save_path = args.output_dir
    cfg.num_classes = args.num_classes
    cfg.score_key = args.score_key

    report = Report()

    # Score-wise: detect mislabeled samples
    detector = DetectLabel(cfg, dataset, report=report)
    detector.detect()

    # Feature-wise: compute embedding distance (rare score)
    print("\n[Step 4b] Computing rare scores (embedding distance)...")
    detector_feature = DetectFeature(cfg, dataset, report=report)
    detector_feature.rare_score()

    # Save report
    report_path = os.path.join(args.output_dir, f"{dataset_name}_report.pt")
    torch.save(report, report_path)
    print(f"  Report saved to {report_path}")

    # ----------------------------------------------------------------
    # Step 5: Score curation
    # ----------------------------------------------------------------
    print("\n[Step 5] Running score curation...")

    # Load the JSONL as HuggingFace dataset (needed by score_curating)
    raw_hf_dataset = hf_load_dataset("json", data_files=args.proxy_jsonl)["train"]

    curated_scores = score_curating(
        report, raw_hf_dataset, args.score_key, args.confidence_prob
    )

    # Compute diversity-based final scores
    rare_samples_info = report.detection['rare_example'][
        : len(report.detection['rare_example']) // 2
    ]
    diversity_scores = np.array(rare_samples_info)[:, 1].tolist()
    final_scores_raw = [
        cur + div for cur, div in zip(curated_scores, diversity_scores)
    ]
    final_scores = [
        max(0, min(args.num_classes - 1, round(s))) for s in final_scores_raw
    ]

    # ----------------------------------------------------------------
    # Step 6: Evaluate AFTER denoise
    # ----------------------------------------------------------------
    denoised_metrics = evaluate(curated_scores, gt_labels, args.num_classes,
                                tag="AFTER denoise (curated_score)")
    final_metrics = evaluate(final_scores, gt_labels, args.num_classes,
                             tag="AFTER denoise (final_curated_score)")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    n_changed = int(sum(1 for a, b in zip(proxy_labels, curated_scores) if a != b))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'Metric':<20} {'Before':>10} {'After(curated)':>15} {'After(final)':>15}")
    print(f"  {'accuracy':<20} {baseline_metrics['accuracy']:>10.4f} "
          f"{denoised_metrics['accuracy']:>15.4f} {final_metrics['accuracy']:>15.4f}")
    print(f"  {'macro_f1':<20} {baseline_metrics['macro_f1']:>10.4f} "
          f"{denoised_metrics['macro_f1']:>15.4f} {final_metrics['macro_f1']:>15.4f}")
    print(f"  {'MAE':<20} {baseline_metrics['mean_abs_error']:>10.4f} "
          f"{denoised_metrics['mean_abs_error']:>15.4f} {final_metrics['mean_abs_error']:>15.4f}")
    print(f"\n  Labels changed: {n_changed} / {n_samples} ({100*n_changed/n_samples:.1f}%)")

    if hasattr(report, "diagnose") and "T" in report.diagnose:
        T = report.diagnose["T"]
        print(f"\n  Transition matrix:\n{np.round(T.numpy() if isinstance(T, torch.Tensor) else T, 3)}")

    # Save metrics
    results = {
        "proxy_jsonl": args.proxy_jsonl,
        "embedding_npy": args.embedding_npy,
        "num_samples": n_samples,
        "num_classes": args.num_classes,
        "confidence_prob": args.confidence_prob,
        "labels_changed": n_changed,
        "labels_changed_pct": round(100 * n_changed / n_samples, 2),
        "baseline_metrics": baseline_metrics,
        "denoised_metrics": denoised_metrics,
        "final_metrics": final_metrics,
    }
    metrics_path = os.path.join(args.output_dir, "denoise_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
