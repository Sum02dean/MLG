import numpy
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm
from data_loader import get_train_chr, filter_genes_by_chr, load_train_genes
from histone_loader import HISTONE_MODS, get_bw_data


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

    def __len__(self) -> int:
        return len(self.genes)

    def __getitem__(self, idx) -> (np.ndarray, np.ndarray):
        gene = self.genes.iloc[idx, :]
        start = gene.TSS_start - self.left_flank_size
        end = gene.TSS_start + self.right_flank_size - 1  # marks last nucleotide index

        features = get_bw_data(gene.cell_line, gene.chr, start, end,
                               histones=self.histone_mods, value_type='mean', n_bins=self.n_bins)
        return np.array(features), gene.gex


def example_train_valid_split():
    chr = numpy.array(get_train_chr())
    chr_train, chr_valid = train_test_split(
        chr, test_size=0.2, random_state=42)

    all_genes = load_train_genes()
    train_genes = filter_genes_by_chr(all_genes, chr_train)
    train_dataloader = torch.utils.data.DataLoader(
        SeqHistDataset(train_genes), shuffle=True, batch_size=16)
    valid_genes = filter_genes_by_chr(all_genes, chr_valid)
    valid_dataloader = torch.utils.data.DataLoader(
        SeqHistDataset(valid_genes), shuffle=True, batch_size=16)

    for gene_features, gex in tqdm(train_dataloader):
        # print(gene_features.shape, gex.shape)
        # ...
        pass


if __name__ == '__main__':
    example_train_valid_split()
