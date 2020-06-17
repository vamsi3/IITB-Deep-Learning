## AUTHOR: Vamsi Krishna Reddy Satti

# ====================================================================================================================
#                                               utils.py
# ====================================================================================================================


import numpy as np
import torch
from torch import nn


def get_model():
    model = nn.Sequential(
        nn.Linear(784, 500),
        nn.ReLU(),
        nn.Linear(500, 300),
        nn.ReLU(),
        nn.Linear(300, 4),
        nn.Softmax(dim=1)
    )
    
    return model


def load_dataset(path):
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    labels, features = data[:, :1], data[:, 1:]
    labels = labels.reshape(-1).astype(np.long)
    labels -= labels.min()

    features -= features.min(0, keepdims=True)
    features_max = features.max(0, keepdims=True)
    features = np.divide(features, features_max, out=np.zeros_like(features), where=(features_max != 0))

    features = torch.from_numpy(features).float()
    labels = torch.from_numpy(labels).long()

    indices = torch.randperm(features.shape[0])
    features, labels = features[indices], labels[indices]
    dataset = torch.utils.data.TensorDataset(features, labels)

    return dataset
