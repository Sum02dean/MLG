import os

import pandas as pd

TRAIN_INFO = {1: ['X1_train_info', 'X1_val_info'], 2: ['X2_train_info', 'X2_val_info']}
TRAIN_LABELS = {1: ['X1_train_y', 'X1_val_y'], 2: ['X2_train_y', 'X2_val_y']}


def load_info(filename: str) -> pd.DataFrame:
    file = f'data{os.sep}CAGE-train{os.sep}{filename}.tsv'
    if not os.path.exists(file):
        file = f'..{os.sep}' + file
    return pd.read_csv(file, sep='\t')


def load_train_genes_for_cell_line(cell_line: int) -> pd.DataFrame:
    """
    Load the train dataset (with expression values) for given cell line.

    :param cell_line: 1 or 2 (corresponds to X1 and X2 datasets)
    :return: DataFrame of gene info and y values as 'gex'
    """
    gene_info = pd.concat([load_info(file) for file in TRAIN_INFO[cell_line]])
    gene_exp = pd.concat([load_info(file) for file in TRAIN_LABELS[cell_line]])

    # verify that the order of genes matches before adding column for expression
    assert (gene_info.gene_name == gene_exp.gene_name).all()
    gene_info['gex'] = gene_exp.gex

    # remove chr prefix from entries
    gene_info['chr'] = gene_info['chr'].map(lambda x: int(x.lstrip('chr')))

    return gene_info


def load_train_genes() -> pd.DataFrame:
    """
    Load the entire train dataset (with expression values).
    Added value of cell gene cell line (1 or 2).

    :return: DataFrame of gene info, cell line number, and y values as 'gex'
    """
    gene_dfs = []
    for cell_line in [1, 2]:
        df = load_train_genes_for_cell_line(cell_line)
        df['cell_line'] = cell_line
        gene_dfs.append(df)

    return pd.concat(gene_dfs)


def load_test_genes() -> pd.DataFrame:
    """
    Load the test dataset.

    :return: DataFrame of gene info for cell line 3
    """
    return load_info('X3_test_info')


def get_train_chr() -> list:
    return list(set(load_train_genes().chr))


def filter_genes_by_chr(genes: pd.DataFrame, chromosomes: list[int]) -> pd.DataFrame:
    """
    Filter genes by chromosome number. Intended for use during cross validation.

    :param genes: DataFrame of gene info
    :param chromosomes: list of chromosomes to filter genes by
    :return: genes from specified chromosomes as a DataFrame
    """
    return genes[genes.chr.isin(chromosomes)]


if __name__ == '__main__':
    all_genes = load_train_genes()
    genes = filter_genes_by_chr(all_genes, get_train_chr()[:5])
    print(genes.head())

    print('average tss length: ', (all_genes.TSS_end - all_genes.TSS_start).mean())