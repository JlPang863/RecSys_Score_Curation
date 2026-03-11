import sys
import os
o_path = os.getcwd()
sys.path.append(o_path) # set path so that modules from other foloders can be loaded

import torch
import numpy as np
import argparse

from docta.utils.config import Config
from docta.datasets import RecSysDataset, CustomizedDataset
from docta.core.preprocess import Preprocess
from docta.apis import DetectLabel, DetectFeature
from docta.core.report import Report
torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
    parser = argparse.ArgumentParser(description='Train a classifier')
    parser.add_argument('--config', help='train config file path', default='template.py')
    parser.add_argument('--dataset_name', help='dataset name for score curation', default='utilitarian')
    parser.add_argument('--dataset_path', help='raw dataset path', default='utilitarian.json')
    parser.add_argument('--feature_key', help='feature keyword in the raw dataset', default='embed_text')
    parser.add_argument('--score_key', help='score keyword in the raw dataset needed to be curated', default='bin_score')
    parser.add_argument('--num_classes', help='the number of score classification used', default=None)

    parser.add_argument('--output_dir', help='output dir', default='results')
    args = parser.parse_args()
    return args


def run_diagnose(
    cfg,
    dataset_name: str,
    dataset_path: str,
    output_dir: str,
    num_classes: int | None = None,
    feature_key: str | None = None,
    score_key: str | None = None,
    embedding_path: str | None = None,
):
    # config setup
    cfg.dataset_type = dataset_name
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.save_path = output_dir
    cfg.preprocessed_dataset_path = (
        os.path.join(output_dir, f"dataset_{dataset_name}.pt")
    )
    if feature_key is not None:
        cfg.feature_key = feature_key
    if score_key is not None:
        cfg.score_key = score_key
        
    if num_classes is not None:
        cfg.num_classes = num_classes

    '''preprocess data'''
    if embedding_path is not None:
        # Use pre-computed embeddings directly (no GPU, no dataset_path needed for diagnosis)
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
        embedding_file = embedding_path
    else:
        # Need raw dataset for encoding
        args_like = argparse.Namespace(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            feature_key=cfg.feature_key,
            score_key=cfg.score_key,
            output_dir=output_dir,
            num_classes=cfg.num_classes,
        )
        dataset = RecSysDataset(cfg, args_like, split="train")
        pre_processor = Preprocess(cfg, dataset)
        pre_processor.encode_feature()
        embedding_file = os.path.join(cfg.save_path, f'embedded_{cfg.dataset_type}.pt')

    # Load embedding file and duplicate for detection
    print(f"[Pipeline] Loading embeddings from {embedding_file}")
    loaded = torch.load(embedding_file)
    n = len(loaded.feature)
    dataset = CustomizedDataset(
        feature=np.concatenate([loaded.feature, loaded.feature]),
        label=np.concatenate([loaded.label, loaded.label]),
        index=np.concatenate([np.arange(n), np.arange(n)]),
    )
    print(f"#Samples (dataset-train) {n}.")


    '''detect data'''
    report = Report()

    #score-wise: score curation technique
    detector = DetectLabel(cfg, dataset, report = report)
    detector.detect()


    ## feature-wise: embedding distance
    print("starting feature-wise part: calculating embedding distance!!!")
    detector_feature = DetectFeature(cfg, dataset, report = report)
    detector_feature.rare_score()


    ##store reports
    report_path = os.path.join(cfg.save_path, f'{cfg.dataset_type}_report.pt')
    torch.save(report, report_path)
    print(f'Report saved to {report_path}')

    return report

if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    
    run_diagnose(
        cfg,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        feature_key=args.feature_key,
        score_key=args.score_key,
    )
