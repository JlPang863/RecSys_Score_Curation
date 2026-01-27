import torch 
from collections import Counter
import random
from datasets import load_dataset
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

seed=3
random.seed(seed)
np.random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Score Curation Process')
    parser.add_argument('--dataset_name', help='dataset name for score curation', default='utilitarian')
    parser.add_argument('--dataset_path', help='raw dataset path', default='utilitarian.json')
    parser.add_argument('--score_key', help='score keyword in the raw dataset needed to be curated', default='bin_score')
    parser.add_argument('--num_classes', help='the number of score classification used', default=None)
    parser.add_argument('--confidence_prob', help='the confidence probability of identifying mis-rated sample', type=float, default=0.5)
    parser.add_argument('--output_dir', help='output dir', default='score_curation_results')
    args = parser.parse_args()
    return args


def score_curating(reports, raw_dataset, score_key, confidence_prob):
    corrupted_samples = [x[0] for x in reports.detection['score_error']]

    curated_sample = []
    curated_sample_scores = []
    for sample in reports.curation['score_curation']:  # (idx, score, confidence)
        if sample[2] >= confidence_prob:  
            curated_sample.append(sample[0])
            curated_sample_scores.append((int(sample[0]), int(sample[1]), round(sample[2], 2)))

    print(f"Curated sample size: {len(curated_sample_scores)}")

    # Filter out some cured samples from corrupted instances
    curated_sample_set = set(curated_sample)
    corrupted_samples_total = [x for x in corrupted_samples if x not in curated_sample_set]

    print(f"Corrupted samples total: {len(corrupted_samples_total)}")

    # Change the original scores to the suggested score
    # scores = torch.load(score_path + "output_scores_revised.pt")
    scores = raw_dataset[score_key][:]
    for sample_score in curated_sample_scores:
        scores[sample_score[0]] = sample_score[1]
        
    return scores



def print_score_heatmap(reports, dataset_name, save_path="figures/"):
    
    data = reports.diagnose['T']
    plt.figure(figsize=(8, 6))
    sns.heatmap(data, annot=True, fmt=".2f", cmap="YlGnBu")

    plt.title(f'Score transition matrix ({dataset_name})', fontsize=18)
    plt.xlabel('Scores', fontsize=18)
    plt.ylabel('Scores', fontsize=18)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    plt.savefig(save_path + f"{dataset_name}_heatmap.pdf", format="pdf", bbox_inches="tight")
    
    
def dataset_load(dataset_path, split='train'):
    lower = dataset_path.lower()
    if lower.endswith(".parquet"):
        raw_dataset = load_dataset("parquet", data_files=dataset_path)[split]
    elif lower.endswith(".jsonl"):
        # json loader supports jsonl
        raw_dataset = load_dataset("json", data_files=dataset_path)[split]
    elif lower.endswith(".json"):
        raw_dataset =  load_dataset("json", data_files=dataset_path)[split]
    else:
        raise ValueError(f"Unsupported file type: {dataset_path}")
        
    return raw_dataset

def run_curation(
    dataset_name: str,
    dataset_path: str,
    output_dir: str,
    score_key: str,
    confidence_prob: float,
):
    """
    Run score curation stage.

    Returns:
        raw_dataset: HuggingFace Dataset with curated scores
    """

    report_path = os.path.join(output_dir, f"{dataset_name}_report.pt")

    # Load raw dataset
    raw_dataset = dataset_load(dataset_path, split="train")

    # Load diagnosis report
    reports = torch.load(report_path)
    print_score_heatmap(reports, dataset_name)

    # Score curation
    curated_scores = score_curating(
        reports,
        raw_dataset,
        score_key=score_key,
        confidence_prob=confidence_prob
    )
    # Diversity score
    rare_samples_info = reports.detection['rare_example'][
        : len(reports.detection['rare_example']) // 2
    ]
    diversity_scores = np.array(rare_samples_info)[:, 1].tolist()

    final_curated_scores = [
        cur + div for cur, div in zip(curated_scores, diversity_scores)
    ]

    # Write back to dataset
    raw_dataset = raw_dataset.add_column("diversity_score", diversity_scores)
    raw_dataset = raw_dataset.add_column("curated_score", curated_scores)
    raw_dataset = raw_dataset.add_column("final_curated_score", final_curated_scores)

    # Save dataset
    final_dataset_path = os.path.join(
        output_dir, f"{dataset_name}_curated.json"
    )
    raw_dataset.to_json(final_dataset_path)

    print(f"Final dataset is saved to {final_dataset_path}")
    
    return raw_dataset

def main(args):
    
    run_curation(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        score_key=args.score_key,
        confidence_prob=args.confidence_prob,
    )

if __name__ == '__main__':
    args = parse_args()
    main(args)
        
    
    