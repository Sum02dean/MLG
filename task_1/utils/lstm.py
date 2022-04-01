import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dataset import *
from stratification import *


# Get fields
train_genes, test_genes = chromosome_splits()
print(train_genes)
n_genes_train, n_features = np.shape(train_genes)
n_genes_test, _ = np.shape(test_genes)
flank_size = 1000
bin_size=100
bin_value_type = 'mean'
histone_mods = 'H3K4me3'

print(np.shape(train_genes))

# # Build Loader
train_dataloader = torch.utils.data.DataLoader(
    HistoneDataset(
        train_genes, histone_mods=histone_mods, bin_value_type=bin_value_type, bin_size=bin_size,
        left_flank_size=flank_size, right_flank_size=flank_size), shuffle=True,
        batch_size=n_genes_train
        )

# Run train loader
for i, (x_train, y_train) in enumerate(train_dataloader):
    print(x_train)
    break
