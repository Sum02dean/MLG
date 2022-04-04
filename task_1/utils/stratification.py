import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split
from utils.data_loader import load_train_genes


def random_splits(test_size=0.3):
    """Generates random splits. Attempts to stratify based on the proportion
       of chromosomes and and cell-lines

    :return: _description_
    :rtype: _type_
    """

    # Pull in data
    df = load_train_genes()
    df.sort_values(by='gene_name', ascending=True, inplace=True)

    # Split on cell lines stratified across chr and
    y_train, y_test = train_test_split(
        df.gex, test_size=test_size,
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
    df = load_train_genes()

    # Split between cell lines
    x1 = df[(df.cell_line == 1)]
    x2 = df[(df.cell_line == 2)]
    return x1, x2


def chromosome_splits(cell_line=None, test_size=0.3, train_cell_line: int = None):
    """Generates splits between chromosomes

    :param cell_line:
        if None: Mutually exclusive splits will be made with cell-line 1 & 2. Cell-line mixing allowed
        if 1: mutually exclusive chr splits will be made across cell-line 1. Cell-line mixing disallowed,
        if 2: mutually exclusive chr splits will be made across cell-line 2 . Cell-line mixing disallowed,
        defaults to None
    :type cell_line: str or None, optional

    :param test_size: ratio to allocate for test size, defaults to 0.3
    :type test_size: float, optional

    :return: train and test data
    :rtype: pandas.core.DataFrame, pandas.core.DataFrame
    """

    # Pull in data
    df = load_train_genes()

    if cell_line is None:
        # Allow mixing between cell-lines
        groups = np.array(df.chr)

    else:
        # Define disjoint selection on cell-lines
        df = df[df.cell_line == cell_line]
        groups = np.array(df.chr)

    # Collect disjoint sets
    gss = GroupShuffleSplit(n_splits=1, train_size=1 - test_size, random_state=42)

    # Get indices
    (train_idx, test_idx) = next(gss.split(X=df, y=None, groups=groups))
    y_train = df.iloc[train_idx]
    y_test = df.iloc[test_idx]
    if train_cell_line is not None:
        assert train_cell_line in [1, 2]
        y_train = y_train[y_train.cell_line == train_cell_line]
        test_cell_line = 2 if train_cell_line == 1 else 1
        y_test = y_test[y_test.cell_line == test_cell_line]

    return y_train, y_test
