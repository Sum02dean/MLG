import os

import pandas as pd

TRAIN_INFO = ['X1_train_info', 'X1_val_info', 'X2_train_info', 'X2_val_info']
TRAIN_LABELS = ['X1_train_y', 'X1_val_y', 'X2_train_y', 'X2_val_y']


def load_info(filename: str) -> pd.DataFrame:
    return pd.read_csv(f'data{os.sep}CAGE-train{os.sep}{filename}.tsv', sep='\t')


def load_train_genes() -> pd.DataFrame:
    """
    Load the entire train dataset (with expression values).

    :return: DataFrame of gene info and y values as 'gex'
    """
    info_dfs = [load_info(file) for file in TRAIN_INFO]
    gene_info = pd.concat(info_dfs)

    label_dfs = [load_info(file) for file in TRAIN_LABELS]
    gene_exp = pd.concat(label_dfs)

    # verify that the order of genes matches before adding column for expression
    assert (gene_info.gene_name == gene_exp.gene_name).all()
    gene_info['gex'] = gene_exp.gex

    return gene_info


def load_test_genes() -> pd.DataFrame:
    return load_info('X3_test_info')


def get_train_chr() -> list:
    return list(set(load_train_genes().chr))


def load_genes_by_chr(chromosomes: list) -> pd.DataFrame:
    """
    Get gene info for only some chromosomes. Intended for use during cross validation.

    :param chromosomes: list of chromosomes to filter genes by
    :return: genes from specified chromosomes as a DataFrame
    """
    # load by chromosomes when using k fold cross validation
    all_genes = load_train_genes()
    return all_genes[all_genes.chr.isin(chromosomes)]


if __name__ == '__main__':
    genes = load_genes_by_chr(get_train_chr()[:5])
