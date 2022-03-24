import os.path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from task_1.utils.data_loader import load_all_genes
from task_1.utils.histone_loader import HISTONE_MODS, get_bw_data
from task_1.utils.stratification import chromosome_split


def get_gene_unique(gene: pd.Series) -> str:
    return f'{gene.cell_line}_{gene.gene_name}'


def generate_histone_pkl(histone_mods: list[str] = None,
                         left_flank_size: int = 1000,
                         right_flank_size: int = 1000,
                         n_bins: int = 20):
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
    df.to_pickle(f'../data/histones_all_l{left_flank_size}_r{right_flank_size}_b{n_bins}.pkl')


class SeqHistDataset(Dataset):

    def __init__(self,
                 genes: pd.DataFrame,
                 histone_mods: list[str] = None,
                 left_flank_size: int = 1000,
                 right_flank_size: int = 1000,
                 bin_size: int = 100) -> None:
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
        start = gene.TSS_start - self.left_flank_size
        end = gene.TSS_start + self.right_flank_size - 1  # marks last nucleotide index

        features = np.array(self.histones[get_gene_unique(gene)])
        # idk why simply to_numpy() couldn't proccess inner lists..
        features = np.array([np.array(x) for x in features])
        if 'gex' not in gene:
            return features
        return features, gene.gex

    def load_histone_data(self):
        if not os.path.exists(self.histone_file):
            generate_histone_pkl(self.histone_mods, self.left_flank_size, self.right_flank_size, self.n_bins)
        return pd.read_pickle(self.histone_file)


def example_train_valid_split():
    train_genes, valid_genes = chromosome_split(test_size=0.2)
    train_dataloader = torch.utils.data.DataLoader(SeqHistDataset(train_genes), shuffle=True, batch_size=16)
    # valid_dataloader = torch.utils.data.DataLoader(SeqHistDataset(valid_genes), shuffle=True, batch_size=16)

    for gene_features, gex in tqdm(train_dataloader):
        # print(gene_features.shape, gex.shape)
        # ...
        pass


if __name__ == '__main__':
    example_train_valid_split()
