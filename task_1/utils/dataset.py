import numpy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold

from histone_loader import HISTONE_MODS, get_bw_data
from data_loader import get_train_chr, filter_genes_by_chr, load_train_genes


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

    def __getitem__(self, idx) -> np.ndarray:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        batch_genes = self.genes.iloc[idx, :]
        features = []
        for _, gene in batch_genes.iterrows():
            start = gene.TSS_start - self.left_flank_size
            end = gene.TSS_start + self.right_flank_size - 1  # marks last nucleotide index

            features.append(
                get_bw_data(gene.cell_line, gene.chr, start, end, histones=self.histone_mods, value_type='mean',
                            n_bins=self.n_bins))
        features = np.array(features)
        print(features.shape)
        return features


if __name__ == '__main__':
    num_folds = 6
    k_fold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    chr = numpy.array(get_train_chr())
    all_genes = load_train_genes()
    for train_index, val_index in k_fold.split(chr):
        chr_train, chr_valid = chr[train_index], chr[val_index]

        genes_train = filter_genes_by_chr(all_genes, chr_train)
        genes_valid = filter_genes_by_chr(all_genes, chr_valid)

        dataset = SeqHistDataset(genes_train)

        for i in range(0, len(genes_train) - 16, 16):
            dataset.__getitem__(list(range(i, i + 16)))
