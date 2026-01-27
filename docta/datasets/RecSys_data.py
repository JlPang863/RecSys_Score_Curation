from .customize import CustomizedDataset
import gzip, json, re
import numpy as np
import os
import torch
from datasets import load_dataset
import pandas as pd

class RecSysDataset(CustomizedDataset):
    def __init__(self, cfg, args, split='train'):

        ## dataset 
        lower = args.dataset_path.lower()
        if lower.endswith(".parquet"):
            self.raw_dataset = load_dataset("parquet", data_files=args.dataset_path)[split]
        elif lower.endswith(".jsonl"):
            # json loader supports jsonl
            self.raw_dataset = load_dataset("json", data_files=args.dataset_path)[split]
        elif lower.endswith(".json"):
            self.raw_dataset =  load_dataset("json", data_files=args.dataset_path)[split]
        else:
            raise ValueError(f"Unsupported file type: {args.dataset_path}")
        

        ###########################################################
        # load & save datasets
        os.makedirs(cfg.save_path, exist_ok=True)
        features, scores = self.load_data_info(args, self.raw_dataset)

        torch.save({'feature': features, 'label': scores}, cfg.preprocessed_dataset_path)
        print(f'Whole dataset size: {len(features)}')
        print(f'Saved preprocessed dataset to {cfg.preprocessed_dataset_path}')

        assert len(features) == len(scores)
        index = range(len(features))
        
        super(RecSysDataset, self).__init__(features, scores, index=index, preprocess=None)
                
                
    def load_data_info(self, args, dataset):
        features = dataset[args.feature_keywords]
        scores = dataset[args.score_keywords]

        return features, np.array(scores)


    
