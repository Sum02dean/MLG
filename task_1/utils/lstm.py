import torch
import torch.nn as nn
import numpy as np
from dataset import *
from stratification import *
from torch.autograd import Variable


class cat_dataloaders():
    def __init__(self, dataloaders):
        """Class to concatenate multiple dataloaders for simultaneous loading.

        :param dataloaders: lsit of data loaders
        :type dataloaders: list
        """
        self.dataloaders = dataloaders
        len(self.dataloaders)

    def __iter__(self):
        self.loader_iter = []
        for data_loader in self.dataloaders:
            self.loader_iter.append(iter(data_loader))
        return self

    def __next__(self):
        out = []
        for data_iter in self.loader_iter:
            out.append(next(data_iter)) # may raise StopIteration
        return tuple(out)

class BasicRNN(nn.Module):
    """ LSTM models """
    def __init__(self, input_size, output_size=20, hidden_size=100, num_layers=1):
        """RNN using LSTM units to predict histone marks for single track.

        :param input_size: This corresponds to the OHE_dim
        :type input_size: int

        :param output_size: This corresponds to n_bins, defaults to 20
        :type output_size: int, optional
        
        :param hidden_size: The number of neurons in the LSTM hidden layers, defaults to 100
        :type hidden_size: int, optional
        
        :param num_layers: The number of LSTM layers, defaults to 1
        :type num_layers: int, optional
        """
        super(BasicRNN, self).__init__()
        
        # Define fields
        self.input_size = input_size # number of features in a sequence vector (OHE_dim)
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM unit
        self.lstm = nn.LSTM(
            input_size=self.input_size, hidden_size=self.hidden_size,
            num_layers=self.num_layers, batch_first=True)

        # FC layer (multiply by num_layers as each lstm layer a returns h-state which I will reshape to a row vector)
        self.fc = nn.Linear(
            in_features=self.hidden_size * self.num_layers, out_features=self.output_size) 
        
        # Activations
        self.relu = nn.ReLU()

    def forward(self, x, batch_size):
        """input shape should have shape (batch_size, seq_len, OHE_dim)"""

        # Initialize hidden and cell states
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).type(dtype))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).type(dtype))

        # LSTM 
        _, (h, c) = self.lstm(x, (h_0, c_0)) # --> (num_layers, batch_size, hidden_size)
        x = h.reshape(batch_size, -1).squeeze() # --> (batch_size, num_layers * hidden_size)
        
        # Linear 
        x = self.relu(self.fc(x)) # --> (batch_size, n_bins)
        x = x.unsqueeze(1) # --> (batch_size, 1, n_bins)
        return x


if __name__ == '__main__':

    # General fields
    train_genes, test_genes = chromosome_splits()
    n_genes_train, n_features = np.shape(train_genes)
    n_genes_test, _ = np.shape(test_genes)
    flank_size = 1000
    bin_size=100
    n_bins = 20
    bin_value_type = 'mean'
    histone_mods = ['H3K4me3']

    # Model fields
    batch_size = 15
    hidden_size = 100
    n_layers = 1
    OHE_dim = 4
    dtype = torch.float


    # Model setup
    model = BasicRNN(input_size=OHE_dim, output_size=n_bins, hidden_size=hidden_size, num_layers=n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
    criterion = nn.MSELoss()
    print(model)

    # Build train loader
    train_dataloader = torch.utils.data.DataLoader(
        HistoneDataset(
            train_genes, histone_mods=histone_mods, bin_value_type=bin_value_type, bin_size=bin_size,
            left_flank_size=flank_size, right_flank_size=flank_size, use_seq=True), shuffle=False,
            batch_size=batch_size
            )

    # Build test loader
    test_dataloader = torch.utils.data.DataLoader(
        HistoneDataset(
            test_genes, histone_mods=histone_mods, bin_value_type=bin_value_type, bin_size=bin_size,
            left_flank_size=flank_size, right_flank_size=flank_size, use_seq=True), shuffle=False,
            batch_size=batch_size
            )

    # seq_data should be (batch_size, seq_len, n_features) or (batch_size, 2000, 4)
    print_every = 10
    for epoch in tqdm(range(1)):
        running_loss = 0.0
        for i, ((gene_features, seq_data), gex_train) in enumerate(train_dataloader):

            # Format dtypes
            gene_features =gene_features.type(dtype)        
            seq_data = seq_data.type(dtype)

        # Forward 
            optimizer.zero_grad()
            output = model(seq_data, batch_size=batch_size)
            assert(output.shape == gene_features.shape)

            # Backward + optimize
            loss = criterion(output, gene_features)
            running_loss += loss.item()
            loss.backward()
            print(output.shape)
            optimizer.step()

            # print every 500 mini-batches
            if i % print_every == print_every-1:    
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_every:.3f}')
                running_loss = 0.0
        
    print('Finished Training')