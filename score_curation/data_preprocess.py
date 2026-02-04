import json
import math
from typing import List, Optional


def preprocess_dataset(
    input_path: str,
    output_path: str,
    num_classes: int,
    score_key: str = "label_score",
    feature_keys: List[str] = None,
    output_score_key: str = "bin_score",
    output_feature_key: str = "embed_text",
    feature_separator: str = " ",
):
    """
    Preprocess dataset for ScoreCurationPipeline.

    1. Discretize continuous scores into bins (bin_score)
    2. Concatenate multiple feature fields into a single embed_text field

    Args:
        input_path: Path to the input JSON file
        output_path: Path to save the processed JSON file
        num_classes: Number of bins for score discretization (e.g., 6 -> bin_score 0-5)
        score_key: Original score field name in the dataset (default: "label_score")
        feature_keys: List of feature fields to concatenate (default: ["history", "label_title"])
        output_score_key: Output field name for discretized score (default: "bin_score")
        output_feature_key: Output field name for concatenated features (default: "embed_text")
        feature_separator: Separator for concatenating features (default: " ")

    Returns:
        data: Processed dataset as a list of dicts
    """

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for sample in data:
        # Discretize score: map [0, 1] -> [0, num_classes-1]
        score = sample[score_key]
        bin_score = int(max(0, min(num_classes - 1, math.ceil(score * num_classes) - 1)))
        sample[output_score_key] = bin_score

        # Concatenate feature fields
        feature_values = [str(sample.get(key, "")) for key in feature_keys]
        sample[output_feature_key] = feature_separator.join(feature_values)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Processed {len(data)} samples")
    print(f"  - {score_key} -> {output_score_key} (num_classes={num_classes})")
    print(f"  - {feature_keys} -> {output_feature_key}")
    print(f"  - Saved to {output_path}")

    return data


if __name__ == "__main__":
    # Example usage
    preprocess_dataset(
        input_path="raw_data/timeline_label.json",
        output_path="raw_data/timeline_label_processed.json",
        num_classes=6,
        score_key="label_score",
        feature_keys=["history", "label_title"],
    )
