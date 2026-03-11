import torch
import numpy as np
from docta.datasets import CustomizedDataset
from .core_utils import mean_pooling
import torch.nn.functional as F
import os
from tqdm import tqdm


def build_dataloader(cfg_loader, dataset):
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg_loader.batch_size,
        num_workers=cfg_loader.num_workers,
        shuffle=cfg_loader.shuffle,
    )


def extract_embedding(cfg, encoder, dataset_list):
    os.makedirs(cfg.save_path, exist_ok=True)
    split_names = ['train', 'test'] if len(dataset_list) > 1 else ['']
    save_paths = []
    model, tokenizer = encoder
    for dataset, split in zip(dataset_list, split_names):
        train_loader = build_dataloader(cfg.embedding_cfg, dataset)
        dataset_embedding, dataset_idx, dataset_label = [], [], []
        for i, batch in tqdm(enumerate(train_loader)):
            batch_feature = batch[0]
            encoded_input = tokenizer(batch_feature).to(cfg.device)
            with torch.no_grad(), torch.amp.autocast(device_type=cfg.device.type):
                model_output = model(**encoded_input)
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            embedding = F.normalize(sentence_embeddings, p=2, dim=1)

            dataset_embedding.append(embedding.cpu().numpy())
            if isinstance(batch[1], list):
                dataset_label.append(torch.stack(batch[1]).cpu().numpy().transpose())
            else:
                dataset_label.append(batch[1].cpu().numpy())
            dataset_idx.append(batch[2])

        # Save all embeddings as a single file
        if len(dataset_label) > 0:
            concat_label = np.concatenate(dataset_label)
            concat_label = concat_label[:, cfg.train_label_sel] if len(concat_label.shape) > 1 else concat_label
            embedded_dataset = CustomizedDataset(
                feature=np.concatenate(dataset_embedding),
                label=concat_label,
                index=np.concatenate(dataset_idx),
            )
            suffix = f'_{split}' if split else ''
            save_path = os.path.join(cfg.save_path, f'embedded_{cfg.dataset_type}{suffix}.pt')
            torch.save(embedded_dataset, save_path)
            save_paths.append(save_path)
            print(f'Save {len(dataset_idx)} batches ({len(embedded_dataset)} samples) to {save_path}')

    return save_paths


class Preprocess:

    def __init__(self, cfg, dataset, test_dataset=None) -> None:
        self.cfg = cfg
        self.dataset = dataset
        self.test_dataset = test_dataset

    def get_encoder(self):
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.embedding_model)
        model = AutoModel.from_pretrained(self.cfg.embedding_model).to(self.cfg.device)

        def tokenize(sentences):
            return tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        return (model, tokenize)

    def encode_feature(self):
        encoder = self.get_encoder()
        dataset_list = [CustomizedDataset(feature=self.dataset.feature, label=self.dataset.label)]
        if self.test_dataset is not None:
            dataset_list.append(CustomizedDataset(feature=self.test_dataset.feature, label=self.test_dataset.label))
        self.save_paths = extract_embedding(self.cfg, encoder, dataset_list)

    def preprocess_rare_pattern(self):
        cfg = self.cfg
        if self.test_dataset is not None:
            print('Ignore test_dataset')
        if cfg.feature_type == 'embedding':
            self.encode_feature()
            print(self.save_paths)
            dataset = torch.load(self.save_paths[0])
            data = dataset.feature
        else:
            raise NotImplementedError(f'feature_type {cfg.feature_type} not defined.')

        self.data_rare_pattern = data
