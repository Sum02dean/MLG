import os
import sys
sys.path.append('../')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import *
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split


def random_splits():
    """Generates random splits. Attempts to stratify based on the proportion
       of chromosomes and and cell-lines

    :return: _description_
    :rtype: _type_
    """

    # Pull in data
    train_chr = get_train_chr()
    df = load_genes_by_chr(train_chr)
    df.sort_values(by='gene_name', ascending=True, inplace=True)

    # Split on cell lines stratified across chr and
    y_train, y_test = train_test_split(
        df.gex, test_size=0.3,
        random_state=0,
        stratify=pd.concat([df.chr, df.cell_line],
                           axis=1))

    # Grab observations
    train_df = df.loc[y_train.index]
    test_df = df.loc[y_test.index]

    return train_df, test_df


def cell_line_splits():
    """simply splits data into X1 and X2

    :return: train and test data, X1 and X2 respectively
    :rtype: pandas.core.DataFrame, pandas.core.DataFrame 
    """

    # Pull in data
    train_chr = get_train_chr()
    df = load_genes_by_chr(train_chr)

    # Split between cell lines
    x1 = df[(df.cell_line == 1)]
    x2 = df[(df.cell_line == 2)]
    return x1, x2


def chromosome_split(cell_line=None, test_size=0.3):
    """Generates splits between chromosomes

    :param cell_line: 
        if None: Mutually exlsive splits will be made with cell-line 1 & 2 mixing allowed
        if 1: mutually exclusive chr splits will be made across cell-line 1 disallowed,
        if 2: mutually exclusive chr splits will be made across cell-line 2 disallowed,
        defaults to None
    :type cell_line: str or None, optional

    :param test_size: ratio to allocate for test size, defaults to 0.3
    :type test_size: float, optional

    :return: train and test data 
    :rtype: pandas.core.DataFrame, pandas.core.DataFrame
    """

    # Pull in data
    train_chr = get_train_chr()
    df = load_genes_by_chr(train_chr)

    if cell_line == None:
        # Allow mixing between cell-lines
        groups = np.array(df.chr)

    else:
        # Define disjoint selection on cell-lines
        df = df[df.cell_line == cell_line]
        groups = np.array(df.chr)

    # Collect disjoin sets
    gss = GroupShuffleSplit(n_splits=1, train_size=1 - test_size)
    for train_idx, test_idx in gss.split(X=df, y=None, groups=groups):
        y_train = df.iloc[train_idx]
        y_test = df.iloc[test_idx]

    return y_train, y_test
