import config as cfg
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class IMDBDataset(Dataset):
    def __init__(self):
        super().__init__()
        data = pd.read_csv(cfg.DATA_CSV)
        self.X = data['review']
        self.y = data['sentiment']
        self.X = self.X.values.tolist()
        self.y = pd.get_dummies(self.y).values.tolist()
        self.tokenizer = BertTokenizer.from_pretrained(cfg.PRETRAINED_WEIGHTS)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        idx = idx.item()
        ids_review = self.tokenizer.encode(self.X[idx][:cfg.MAX_SEQ_LEN], max_length=cfg.MAX_SEQ_LEN)
        padding = [0] * (cfg.MAX_SEQ_LEN - len(ids_review))
        ids_review += padding
        assert len(ids_review) == cfg.MAX_SEQ_LEN
        ids_review = torch.tensor(ids_review)
        sentiment = self.y[idx]
        sentiment = torch.from_numpy(np.array(sentiment))
        return ids_review, sentiment
