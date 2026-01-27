import sys
import os
o_path = os.getcwd()
sys.path.append(o_path) # set path so that modules from other foloders can be loaded

import torch
import argparse

from docta.utils.config import Config
from docta.datasets import RecSysDataset
from docta.core.preprocess import Preprocess
from docta.datasets.data_utils import load_embedding
torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
    parser = argparse.ArgumentParser(description='Train a classifier')
    parser.add_argument('--config', help='train config file path', default='template.py')
    parser.add_argument('--dataset_name', help='dataset name for score curation', default='utilitarian')
    parser.add_argument('--dataset_path', help='raw dataset path', default='utilitarian.json')
    parser.add_argument('--feature_keywords', help='feature keyword in the raw dataset', default='embed_text')
    parser.add_argument('--score_keywords', help='score keyword in the raw dataset needed to be curated', default='bin_score')

    parser.add_argument('--output_dir', help='output dir', default='score_curation_results/')
    args = parser.parse_args()
    return args



'''load data'''
args = parse_args()
cfg = Config.fromfile(args.config)


#### Required Input
cfg.dataset_type = args.dataset_name
# cfg.feature_keywords = args.feature_keywords
# cfg.score_keywords = args.score_keywords

cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# cfg.data_root = args.score_root_path
cfg.save_path =  args.output_dir 
cfg.preprocessed_dataset_path = cfg.save_path + f'dataset_{args.dataset_name}.pt'


dataset = RecSysDataset(cfg, args, split='train')

print(f'Dataset {args.dataset_name} load finished')


'''preprocess data'''
pre_processor = Preprocess(cfg, dataset)
pre_processor.encode_feature()
print(pre_processor.save_ckpt_idx)


data_path = lambda x: cfg.save_path + f'embedded_{cfg.dataset_type}_{x}.pt'

import pdb;pdb.set_trace()

dataset, _ = load_embedding(pre_processor.save_ckpt_idx, data_path, duplicate=True) ## duplicate dataset


'''detect data'''
from docta.apis import DetectLabel, DetectFeature
from docta.core.report import Report
report = Report()

#score-wise: score curation technique
detector = DetectLabel(cfg, dataset, report = report)
detector.detect()



## feature-wise: embedding distance
print("starting feature-wise part: calculating embedding distance!!!")
detector_feature = DetectFeature(cfg, dataset, report = report)
detector_feature.rare_score()


##store reports
report_path = cfg.save_path + f'{cfg.dataset_type}_report.pt'
torch.save(report, report_path)
print(f'Report saved to {report_path}')

