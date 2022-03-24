import numpy as np
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from data_loader import *
from dataset import *
import torch
from histone_loader import*
from stratification import *

clf = RandomForestRegressor(n_estimators=10)

param_dist = {
    "max_depth": [3, None],
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy"]
}

# Get genes
train_genes, test_genes = random_splits()

train_dataloader = torch.utils.data.DataLoader(
    SeqHistDataset(train_genes), shuffle=True, batch_size=16)

for gene_features, gex in tqdm(train_dataloader):
    print(gene_features.shape, gex.shape)
    break
