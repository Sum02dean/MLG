import os.path
import random

import numpy as np
import pandas as pd
import scipy
import torch
from sklearn import svm, clone
from sklearn.ensemble import RandomForestRegressor

from utils.data_loader import load_train_genes, load_test_genes
from utils.dataset import HistoneDataset
from utils.stratification import chromosome_splits


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def spearman_score(y, y_true):
    return scipy.stats.spearmanr(y, y_true)[0]


def dataloader(genes, histones=None, bin_size=50, flank_size=1000, batch_size=None):
    if batch_size is None:
        batch_size = np.shape(genes)[0]
    if histones is None:
        histones = ['DNase', 'H3K4me1', 'H3K4me3', 'H3K27ac', 'H3K27me3', 'H3K36me3']
    train_dataset = HistoneDataset(genes, bin_size=bin_size, left_flank_size=flank_size,
                                   right_flank_size=flank_size, histone_mods=histones)
    return torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)


def cross_validate(base_model, train_cell_line=1, histones=None, bin_size=50, flank_size=1000,
                   n_folds: int = 4) -> float:
    train_genes, test_genes = chromosome_splits(test_size=0.2, train_cell_line=train_cell_line)
    train_loader = dataloader(train_genes, histones, bin_size, flank_size, np.shape(train_genes)[0] // n_folds)
    test_loader = dataloader(test_genes, histones, bin_size, flank_size, np.shape(test_genes)[0] // n_folds)

    val_scores = 0
    for i, ((x_train, y_train), (x_val, y_val)) in enumerate(zip(train_loader, test_loader)):
        if i == n_folds:
            break
        n_genes_train, n_features, n_bins = x_train.shape
        n_genes_val, _, _ = x_val.shape

        x_train = x_train.reshape(n_genes_train, n_features * n_bins)
        x_val = x_val.reshape(n_genes_val, n_features * n_bins)

        model = clone(base_model)
        model.fit(x_train, y_train)
        y_pred_train, y_pred_val = model.predict(x_train), model.predict(x_val)

        train_score, val_score = spearman_score(y_pred_train, y_train), spearman_score(y_pred_val, y_val)
        # print(f'train {train_score}, val {val_score}')
        val_scores += val_score

    return val_scores / n_folds


def train_and_predict(train_genes: pd.DataFrame, test_genes: pd.DataFrame, model, bin_size=50, flank_size=1000):
    # custom_scorer = make_scorer(spearman_score, greater_is_better=True)
    histones = ['DNase', 'H3K4me1', 'H3K4me3', 'H3K27ac', 'H3K27me3', 'H3K36me3']

    train_dataloader = dataloader(train_genes, histones=histones, bin_size=bin_size, flank_size=flank_size,
                                  batch_size=np.shape(train_genes)[0])
    test_dataloader = dataloader(test_genes, histones=histones, bin_size=bin_size, flank_size=flank_size,
                                 batch_size=np.shape(test_genes)[0])

    # Run train loader
    (x_train, y_train) = next(iter(train_dataloader))
    n_genes_train, n_features, n_bins = x_train.shape
    x_train = x_train.reshape(n_genes_train, n_features * n_bins)

    # Run test loader
    (x_test, y_test) = next(iter(test_dataloader))
    n_genes_test, _, _ = x_test.shape
    x_test = x_test.reshape(n_genes_test, n_features * n_bins)

    # Fit train data
    print('Fitting...')
    model.fit(x_train, y_train)

    # Predict test
    print('Predicting...')
    y_pred = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    train_score = spearman_score(y_pred_train, y_train)
    print(f'Spearman Correlation Score train: {train_score}')
    if y_test.nelement() != 0:
        test_score = spearman_score(y_pred, y_test)
        print(f'Spearman Correlation Score test: {test_score}')
    return y_pred


def rf_model(max_depth=20, n_estimators=30, bootstrap=True):
    return RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, bootstrap=bootstrap, n_jobs=-1,
                                 random_state=42)


def svm_model(kernel: str = 'rbf', c: int = 1):
    return svm.SVR(kernel=kernel, C=c)


def create_submission(test_genes: pd.DataFrame, pred: np.array) -> None:
    save_dir = '../data/submissions'
    file_name = 'gex_predicted.csv'  # DO NOT CHANGE THIS
    zip_name = "Kasak_Liine_Project1.zip"
    save_path = f'{save_dir}/{zip_name}'
    compression_options = dict(method="zip", archive_name=file_name)

    test_genes['gex_predicted'] = pred.tolist()
    print(f'Saving submission to path {os.path.abspath(save_dir)}')
    test_genes[['gene_name', 'gex_predicted']].to_csv(save_path, compression=compression_options)


def best_rf():
    train_genes, test_genes = chromosome_splits(test_size=0.2)
    train_and_predict(train_genes, test_genes, rf_model(), flank_size=500, bin_size=1000)
    print(cross_validate(rf_model(), train_cell_line=1, flank_size=500, bin_size=1000))
    print(cross_validate(rf_model(), train_cell_line=1, flank_size=500, bin_size=1000))

    train, test = load_train_genes(), load_test_genes()
    y_pred = train_and_predict(train, test, rf_model(), bin_size=500, flank_size=1000)
    create_submission(test_genes=test, pred=y_pred)


if __name__ == '__main__':
    set_seed()
    best_rf()
    # cross_validate(rf_model())
