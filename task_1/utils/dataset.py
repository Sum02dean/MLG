import os.path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from data_loader import load_all_genes
from histone_loader import HISTONE_MODS, get_bw_data, str_to_idx
from stratification import chromosome_splits
from seq_loader import get_encoded_sequence


def get_gene_unique(gene: pd.Series) -> str:
    """
    Returns a unique string representation for given gene information.

    :param gene: Series object including cell_line and gene_name.
    :return: string representing given gene
    """
    return f'{gene.cell_line}_{gene.gene_name}'


def get_histone_filename(left_flank_size: int, right_flank_size: int, n_bins: int, bin_value_type: str) -> str:
    return f'../data/histones_l{left_flank_size}_r{right_flank_size}_b{n_bins}_{bin_value_type}.pkl'


def get_seq_filename(left_flank_size: int, right_flank_size: int) -> str:
    return f'../data/seq_l{left_flank_size}_r{right_flank_size}.pkl'


def get_data(data_type: str, left_flank_size: int, right_flank_size: int, n_bins: int = 100,
             bin_value_type: str = 'mean') -> pd.DataFrame:
    """
    Generates histone modification data by bins for each gene.

    :param data_type: 'seq' or 'histone' data
    :param left_flank_size: number of nucleotides to the left side of TSS start to look at
    :param right_flank_size: number of nucleotides to the right side of TSS start to look at (including TSS_start)
    :param n_bins: number of bins to average histone modification signal over sequence
    :param bin_value_type: method how to average bin values
    """
    assert data_type in ['seq', 'histone']
    print(f'Generating pkl file with {data_type} data...')
    all_genes = load_all_genes()
    data_per_gene = {}
    for i in tqdm(range(len(all_genes))):
        gene = all_genes.iloc[i, :]
        start = gene.TSS_start - left_flank_size
        end = gene.TSS_start + right_flank_size - 1  # marks last nucleotide index

        if data_type == 'histone':
            features = get_bw_data(gene.cell_line, gene.chr, start, end, value_type=bin_value_type, n_bins=n_bins)
        else:
            features = get_encoded_sequence(gene.chr, start, end, gene.strand == '-')
        data_per_gene[get_gene_unique(gene)] = features
    return pd.DataFrame.from_dict(data_per_gene)


def get_seq_data(left_flank_size: int, right_flank_size: int) -> pd.DataFrame:
    """
    Generates DNA seq data for each gene.

    :param left_flank_size: number of nucleotides to the left side of TSS start to look at
    :param right_flank_size: number of nucleotides to the right side of TSS start to look at (including TSS_start)
    """
    print('Generating pkl file with sequence data...')
    all_genes = load_all_genes()
    data_per_gene = {}
    for i in tqdm(range(len(all_genes))):
        gene = all_genes.iloc[i, :]
        start = gene.TSS_start - left_flank_size
        end = gene.TSS_start + right_flank_size - 1  # marks last nucleotide index

        features = get_bw_data(gene.cell_line, gene.chr, start, end, value_type=bin_value_type, n_bins=n_bins)
        data_per_gene[get_gene_unique(gene)] = features
    return pd.DataFrame.from_dict(data_per_gene)


def list_2d_to_np(l: list[list]) -> np.ndarray:
    return np.array([np.array(e) for e in l])


class HistoneDataset(Dataset):

    def __init__(self,
                 genes: pd.DataFrame,
                 use_seq: bool = False,
                 histone_mods: list[str] = None,
                 left_flank_size: int = 1000,
                 right_flank_size: int = 1000,
                 bin_size: int = 100,
                 bin_value_type: str = 'mean') -> None:
        """
        DataSet for model training based on histone modification data alone.
        Load histone modification signal averages or pre-generate if missing.

        :param genes: DataFrame of gene information from CAGE-train, including cell_line and gex for train genes
        :param histone_mods: list of histone modification signal types to look at
        :param left_flank_size: number of nucleotides to the left side of TSS start to look at
        :param right_flank_size: number of nucleotides to the right side of TSS start to look at (including TSS_start)
        :param bin_size: length of sequence to average histone modification values over
        :param bin_value_type: method how to average bin values
        """
        if histone_mods is None:
            histone_mods = HISTONE_MODS

        self.genes = genes
        n_bins = int((left_flank_size + right_flank_size) / bin_size)

        self.histones = HistoneDataset.load_histone_data(histone_mods, left_flank_size, right_flank_size, n_bins,
                                                         bin_value_type)
        self.sequences = None
        if use_seq:
            self.sequences = HistoneDataset.load_seq_data(left_flank_size, right_flank_size)

    def __len__(self) -> int:
        return len(self.genes)

    def __getitem__(self, idx) -> (np.ndarray, np.ndarray):
        gene = self.genes.iloc[idx, :]

        features = self.histones[get_gene_unique(gene)]
        # idk why simply to_numpy() couldn't process inner lists..
        features = list_2d_to_np(features)  # shape (batch_size, nr_histones, nr_bins)
        if self.sequences is not None:
            # seq data shape: (batch_size, left_flank + right flank, 4)
            features = features, list_2d_to_np(self.sequences[get_gene_unique(gene)])
        if 'gex' not in gene:
            return features
        return features, gene.gex

    @staticmethod
    def load_histone_data(histone_mods: list[str], left_flank_size: int, right_flank_size: int, n_bins: int,
                          bin_value_type: str):
        histone_file = get_histone_filename(left_flank_size, right_flank_size, n_bins, bin_value_type)
        if not os.path.exists(histone_file):
            df = get_data('histone', left_flank_size, right_flank_size, n_bins, bin_value_type)
            df.to_pickle(histone_file)
            print(f'Saved histone data to {histone_file}')
        df = pd.read_pickle(histone_file)
        return df.iloc[str_to_idx(histone_mods)]

    @staticmethod
    def load_seq_data(left_flank_size: int, right_flank_size: int):
        seq_file = get_seq_filename(left_flank_size, right_flank_size)
        if not os.path.exists(seq_file):
            df = get_data('seq', left_flank_size, right_flank_size)
            df.to_pickle(seq_file)
            print(f'Saved seq data to {seq_file}')
        return pd.read_pickle(seq_file)

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
        HistoneDataset(train_genes, use_seq=True), shuffle=True, batch_size=16)
    # valid_dataloader = torch.utils.data.DataLoader(SeqHistDataset(valid_genes), shuffle=True, batch_size=16)

    for (gene_features, seq_data), gex in tqdm(train_dataloader):
        print(f'{gene_features.shape} {seq_data.shape}', gex.shape)
        # ...
        break


if __name__ == '__main__':
    example_train_valid_split()
