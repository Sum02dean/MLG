import os.path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from data_loader import load_all_genes
from histone_loader import HISTONE_MODS, get_bw_data
from stratification import chromosome_splits
 

def get_gene_unique(gene: pd.Series) -> str:
    """
    Returns a unique string representation for given gene information.

    :param gene: Series object including cell_line and gene_name.
    :return: string representing given gene
    """
    return f'{gene.cell_line}_{gene.gene_name}'


def generate_histone_pkl(histone_mods: list[str] = None,
                         left_flank_size: int = 1000,
                         right_flank_size: int = 1000,
                         n_bins: int = 20):
    """
    Generates histone modification data by bins for each gene and save to a pickle file.

    :param histone_mods: list of histone modification signal types to look at (NB! currently, only default supported)
    :param left_flank_size: number of nucleotides to the left side of TSS start to look at
    :param right_flank_size: number of nucleotides to the right side of TSS start to look at (including TSS_start)
    :param n_bins: number of bins to average histone modification signal over sequence
    """
    print('Generating pkl file with histone modification data...')
    all_genes = load_all_genes()
    data_per_gene = {}
    for i in tqdm(range(len(all_genes))):
        gene = all_genes.iloc[i, :]
        start = gene.TSS_start - left_flank_size
        end = gene.TSS_start + right_flank_size - 1  # marks last nucleotide index

        features = get_bw_data(gene.cell_line, gene.chr, start, end, histones=histone_mods, value_type='mean',
                               n_bins=n_bins)
        data_per_gene[get_gene_unique(gene)] = features
    df = pd.DataFrame.from_dict(data_per_gene)
    df.to_pickle(
        f'../data/histones_all_l{left_flank_size}_r{right_flank_size}_b{n_bins}.pkl')


class HistoneDataset(Dataset):

    def __init__(self,
                 genes: pd.DataFrame,
                 histone_mods: list[str] = None,
                 left_flank_size: int = 1000,
                 right_flank_size: int = 1000,
                 bin_size: int = 100) -> None:
        """
        DataSet for model training based on histone modification data alone.
        Load histone modification signal averages or pre-generate if missing.

        :param genes: DataFrame of gene information from CAGE-train, including cell_line and gex for train genes
        :param histone_mods: list of histone modification signal types to look at (NB! currently, only default supported)
        :param left_flank_size: number of nucleotides to the left side of TSS start to look at
        :param right_flank_size: number of nucleotides to the right side of TSS start to look at (including TSS_start)
        :param bin_size: length of sequence to average histone modification values over
        """
        if histone_mods is None:
            histone_mods = HISTONE_MODS

        self.genes = genes
        self.histone_mods = histone_mods
        self.left_flank_size = left_flank_size
        self.right_flank_size = right_flank_size
        self.n_bins = int((left_flank_size + right_flank_size) / bin_size)

        self.histone_file = f'../data/histones_all_l{left_flank_size}_r{right_flank_size}_b{self.n_bins}.pkl'
        self.histones = self.load_histone_data()
        pass

    def __len__(self) -> int:
        return len(self.genes)

    def __getitem__(self, idx) -> (np.ndarray, np.ndarray):
        gene = self.genes.iloc[idx, :]

        features = self.histones[get_gene_unique(gene)]
        # idk why simply to_numpy() couldn't proccess inner lists..
        features = np.array([np.array(x) for x in features])
        if 'gex' not in gene:
            return features
        return features, gene.gex

    def load_histone_data(self):
        if not os.path.exists(self.histone_file):
            generate_histone_pkl(
                self.histone_mods, self.left_flank_size, self.right_flank_size, self.n_bins)
        return pd.read_pickle(self.histone_file)

class HistoneDataset_returngenenames(HistoneDataset):
    def __getitem__(self, idx) -> (np.ndarray, np.ndarray):
        gene = self.genes.iloc[idx, :]

        features = self.histones[get_gene_unique(gene)]
        # idk why simply to_numpy() couldn't proccess inner lists..
        features = np.array([np.array(x) for x in features])
        if 'gex' not in gene:
            return features
        return features, gene.gex,get_gene_unique(gene)	



def example_train_valid_split():

    train_genes, valid_genes = chromosome_splits(test_size=0.2)
    train_dataloader = torch.utils.data.DataLoader(
        HistoneDataset(train_genes), shuffle=True, batch_size=16)
    # valid_dataloader = torch.utils.data.DataLoader(SeqHistDataset(valid_genes), shuffle=True, batch_size=16)

    for gene_features, gex in tqdm(train_dataloader):
        # print(gene_features.shape, gex.shape)
        # ...
        pass


if __name__ == '__main__':
    example_train_valid_split()
