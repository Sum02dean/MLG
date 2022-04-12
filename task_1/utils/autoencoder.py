import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import scipy
from dataset import *

# Get genes
train_genes, test_genes = chromosome_splits()
n_genes_train, _ = np.shape(train_genes)
n_genes_test, _ = np.shape(test_genes)
flank_size = 1000
bin_size = 50
histone_mods =['H3K4me3']
bin_value_type = 'mean'
dtype = torch.float
batch_size = 1


# Conv AE
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
       
        #Encoder
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # # Latent
        # self.latent = nn.Linear()
        
        #Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=1)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.t_conv1(x))
        x = F.sigmoid(self.t_conv2(x))
        return x

# Load train data
train_dataloader = torch.utils.data.DataLoader(
    HistoneDataset(train_genes, histone_mods=histone_mods, bin_value_type=bin_value_type, bin_size=bin_size,
                    left_flank_size=flank_size, right_flank_size=flank_size, use_seq=True), shuffle=True,
    batch_size=batch_size)

# Load test data
test_dataloader = torch.utils.data.DataLoader(
    HistoneDataset(test_genes, histone_mods=histone_mods, bin_value_type=bin_value_type, bin_size=bin_size,
                    left_flank_size=flank_size, right_flank_size=flank_size, use_seq=True), shuffle=False,
    batch_size=batch_size)

for i, ((gene_features, seq_data), _) in enumerate(train_dataloader):
   
    # Format dtypes
    gene_features = gene_features.type(dtype)
    seq_data = seq_data.type(dtype).squeeze()
    print(seq_data.shape)

    #Instantiate the model
    model = ConvAutoencoder()
    print(model)
    
    break


























# #Instantiate the model
# model = ConvAutoencoder()
# print(model)