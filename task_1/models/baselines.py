import os.path
import random

import numpy as np
import pandas as pd
import scipy
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer

from utils.data_loader import load_train_genes, load_test_genes
from utils.dataset import HistoneDataset
from utils.histone_loader import HISTONE_MODS
from utils.stratification import chromosome_splits


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def spearman_score(y, y_true):
    return scipy.stats.spearmanr(y, y_true)[0]


def train_and_predict(train_genes: pd.DataFrame, test_genes: pd.DataFrame, model, bin_size=50, flank_size=1000):
    # custom_scorer = make_scorer(spearman_score, greater_is_better=True)
    histones = ['DNase', 'H3K4me1', 'H3K4me3', 'H3K27ac', 'H3K27me3', 'H3K36me3']

    # Load train data
    train_dataset = HistoneDataset(train_genes, bin_size=bin_size, left_flank_size=flank_size,
                                   right_flank_size=flank_size, histone_mods=histones)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=np.shape(train_genes)[0])

    # Load test data
    test_dataset = HistoneDataset(test_genes, bin_size=bin_size, left_flank_size=flank_size,
                                  right_flank_size=flank_size, histone_mods=histones)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=np.shape(test_genes)[0])

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


def rf_model():
    return RandomForestRegressor(max_depth=5, n_estimators=18, bootstrap=True, n_jobs=-1, random_state=42)


def random_forest(histone_mods: list[str] = HISTONE_MODS, bin_value_type: str = 'mean', bin_size: int = 100,
                  flank_size: int = 1000):
    print(f'Random Forest, flank {flank_size}, bin {bin_size}, {bin_value_type} bin values, histones: {histone_mods}')
    rf_params = {
        'max_depth': [30],
        'n_estimators': [300],
        'bootstrap': [True],
    }
    # train_and_cross_validate(RandomForestRegressor(n_jobs=-1, random_state=42), rf_params, histone_mods, bin_value_type,
    #                          bin_size, flank_size)
    # All histones: Spearman correlation score 0.7511


def support_vector_machine(histone_mods: list[str] = HISTONE_MODS, bin_value_type: str = 'mean', bin_size: int = 100,
                           flank_size: int = 1000):
    print(f'SVM, {bin_value_type} bin values, histones:{histone_mods}')
    svm_params = {
        'kernel': ['rbf'],
        'C': [10]}
    # train_and_cross_validate(svm.SVR(), svm_params, histone_mods, bin_value_type, bin_size, flank_size)
    # All histones: Spearman correlation score 0.7328


def create_submission(test_genes: pd.DataFrame, pred: np.array) -> None:
    save_dir = '../data/submissions'
    file_name = 'gex_predicted.csv'  # DO NOT CHANGE THIS
    zip_name = "Kasak_Liine_Project1.zip"
    save_path = f'{save_dir}/{zip_name}'
    compression_options = dict(method="zip", archive_name=file_name)

    test_genes['gex_predicted'] = pred.tolist()
    print(f'Saving submission to path {os.path.abspath(save_dir)}')
    test_genes[['gene_name', 'gex_predicted']].to_csv(save_path, compression=compression_options)


if __name__ == '__main__':
    set_seed()
    train, test = chromosome_splits(test_size=0.2, train_cell_line=1)
    train_and_predict(train, test, rf_model(), bin_size=100, flank_size=1000)
    train, test = chromosome_splits(test_size=0.2, train_cell_line=2)
    train_and_predict(train, test, rf_model(), bin_size=100, flank_size=1000)

    train, test = chromosome_splits(cell_line=1, test_size=0.2)
    train_and_predict(train, test, rf_model(), bin_size=100, flank_size=1000)
    train, test = chromosome_splits(cell_line=2, test_size=0.2)
    train_and_predict(train, test, rf_model(), bin_size=100, flank_size=1000)

    # train, test = load_train_genes(), load_test_genes()
    # y_pred = train_and_predict(train, test,
    #                            RandomForestRegressor(max_depth=30, n_estimators=300, bootstrap=True, n_jobs=-1,
    #                                                  random_state=42), bin_size=100,
    #                            flank_size=1000)
    # create_submission(test_genes=test, pred=y_pred)
