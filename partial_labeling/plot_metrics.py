"""
Plot metrics comparison across different train ratios.
Reads metrics.json from each subdirectory and generates comparison plots.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot metrics comparison across train ratios")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Directory containing subdirectories with metrics.json files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save plots (default: experiment_dir/plots)",
    )
    return parser.parse_args()


def load_metrics(experiment_dir: Path) -> dict:
    """Load all metrics.json files from subdirectories."""
    results = {}

    for subdir in sorted(experiment_dir.iterdir()):
        if not subdir.is_dir():
            continue
        metrics_file = subdir / "metrics.json"
        if not metrics_file.exists():
            continue

        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        train_ratio = metrics.get("train_ratio", None)
        if train_ratio is not None:
            results[train_ratio] = metrics

    return results


def plot_metrics(results: dict, output_dir: Path) -> None:
    """Generate comparison plots for all metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort by train ratio
    train_ratios = sorted(results.keys())

    # Extract metrics
    accuracies = [results[tr]["accuracy"] for tr in train_ratios]
    macro_f1s = [results[tr]["macro_f1"] for tr in train_ratios]
    maes = [results[tr]["mean_abs_error"] for tr in train_ratios]
    train_rows = [results[tr]["train_rows"] for tr in train_ratios]
    test_rows = [results[tr]["test_rows"] for tr in train_ratios]

    # Convert train ratios to percentages for display
    train_ratio_pcts = [tr * 100 for tr in train_ratios]

    # Set style
    plt.style.use("seaborn-v0_8-whitegrid")
    fig_size = (10, 6)

    # ========================================
    # Plot 1: Accuracy vs Train Ratio
    # ========================================
    plt.figure(figsize=fig_size)
    plt.plot(train_ratio_pcts, accuracies, "o-", linewidth=2, markersize=8, color="#2196F3")
    plt.xlabel("Train Ratio (%)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Proxy kNN Accuracy vs Train Ratio", fontsize=14)
    plt.xticks(train_ratio_pcts)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

    # Add value labels
    for x, y in zip(train_ratio_pcts, accuracies):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_train_ratio.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "accuracy_vs_train_ratio.pdf", bbox_inches="tight")
    plt.close()

    # ========================================
    # Plot 2: Macro F1 vs Train Ratio
    # ========================================
    plt.figure(figsize=fig_size)
    plt.plot(train_ratio_pcts, macro_f1s, "s-", linewidth=2, markersize=8, color="#4CAF50")
    plt.xlabel("Train Ratio (%)", fontsize=12)
    plt.ylabel("Macro F1", fontsize=12)
    plt.title("Proxy kNN Macro F1 vs Train Ratio", fontsize=14)
    plt.xticks(train_ratio_pcts)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

    for x, y in zip(train_ratio_pcts, macro_f1s):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "macro_f1_vs_train_ratio.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "macro_f1_vs_train_ratio.pdf", bbox_inches="tight")
    plt.close()

    # ========================================
    # Plot 3: MAE vs Train Ratio
    # ========================================
    plt.figure(figsize=fig_size)
    plt.plot(train_ratio_pcts, maes, "^-", linewidth=2, markersize=8, color="#FF5722")
    plt.xlabel("Train Ratio (%)", fontsize=12)
    plt.ylabel("Mean Absolute Error", fontsize=12)
    plt.title("Proxy kNN MAE vs Train Ratio", fontsize=14)
    plt.xticks(train_ratio_pcts)
    plt.grid(True, alpha=0.3)

    for x, y in zip(train_ratio_pcts, maes):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "mae_vs_train_ratio.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "mae_vs_train_ratio.pdf", bbox_inches="tight")
    plt.close()

    # ========================================
    # Plot 4: All Metrics Combined
    # ========================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Accuracy
    axes[0].plot(train_ratio_pcts, accuracies, "o-", linewidth=2, markersize=8, color="#2196F3")
    axes[0].set_xlabel("Train Ratio (%)", fontsize=11)
    axes[0].set_ylabel("Accuracy", fontsize=11)
    axes[0].set_title("Accuracy", fontsize=12)
    axes[0].set_xticks(train_ratio_pcts)
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)

    # Macro F1
    axes[1].plot(train_ratio_pcts, macro_f1s, "s-", linewidth=2, markersize=8, color="#4CAF50")
    axes[1].set_xlabel("Train Ratio (%)", fontsize=11)
    axes[1].set_ylabel("Macro F1", fontsize=11)
    axes[1].set_title("Macro F1", fontsize=12)
    axes[1].set_xticks(train_ratio_pcts)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    # MAE
    axes[2].plot(train_ratio_pcts, maes, "^-", linewidth=2, markersize=8, color="#FF5722")
    axes[2].set_xlabel("Train Ratio (%)", fontsize=11)
    axes[2].set_ylabel("MAE", fontsize=11)
    axes[2].set_title("Mean Absolute Error", fontsize=12)
    axes[2].set_xticks(train_ratio_pcts)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Proxy kNN Performance vs Train Ratio", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "all_metrics_combined.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "all_metrics_combined.pdf", bbox_inches="tight")
    plt.close()

    # ========================================
    # Plot 5: Accuracy & F1 on same plot
    # ========================================
    plt.figure(figsize=fig_size)
    plt.plot(train_ratio_pcts, accuracies, "o-", linewidth=2, markersize=8, label="Accuracy", color="#2196F3")
    plt.plot(train_ratio_pcts, macro_f1s, "s-", linewidth=2, markersize=8, label="Macro F1", color="#4CAF50")
    plt.xlabel("Train Ratio (%)", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title("Proxy kNN: Accuracy & Macro F1 vs Train Ratio", fontsize=14)
    plt.xticks(train_ratio_pcts)
    plt.ylim(0, 1)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_f1_comparison.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "accuracy_f1_comparison.pdf", bbox_inches="tight")
    plt.close()

    # ========================================
    # Save summary table
    # ========================================
    summary = {
        "train_ratio": train_ratios,
        "train_ratio_pct": train_ratio_pcts,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "accuracy": accuracies,
        "macro_f1": macro_f1s,
        "mean_abs_error": maes,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print summary table
    print("\n" + "=" * 70)
    print("Summary Table")
    print("=" * 70)
    print(f"{'Train Ratio':>12} | {'Train Rows':>10} | {'Test Rows':>10} | {'Accuracy':>10} | {'Macro F1':>10} | {'MAE':>10}")
    print("-" * 70)
    for i, tr in enumerate(train_ratios):
        print(f"{tr*100:>11.1f}% | {train_rows[i]:>10} | {test_rows[i]:>10} | {accuracies[i]:>10.4f} | {macro_f1s[i]:>10.4f} | {maes[i]:>10.4f}")
    print("=" * 70)

    print(f"\nPlots saved to: {output_dir}")
    print(f"Summary saved to: {summary_path}")


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir / "plots"

    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    print(f"Loading metrics from: {experiment_dir}")
    results = load_metrics(experiment_dir)

    if not results:
        raise ValueError(f"No metrics.json files found in subdirectories of {experiment_dir}")

    print(f"Found {len(results)} train ratio results: {sorted(results.keys())}")

    plot_metrics(results, output_dir)


if __name__ == "__main__":
    main()
