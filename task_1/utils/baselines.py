import random

import scipy
from sklearn import svm
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from dataset import *
from stratification import *


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def spearman_scoring(y, y_true):
    return scipy.stats.spearmanr(y, y_true)[0]


def train_and_predict(model: BaseEstimator, params: dict, histone_mods: list[str], bin_value_type: str, bin_size: int,
                      flank_size: int):
    # Load models
    custom_scorer = make_scorer(spearman_scoring, greater_is_better=True)
    clf = GridSearchCV(estimator=model, param_grid=params, cv=3, n_jobs=-1, scoring=custom_scorer)

    # Get genes
    train_genes, test_genes = chromosome_splits()
    n_genes_train, _ = np.shape(train_genes)
    n_genes_test, _ = np.shape(test_genes)

    # Load train data
    train_dataloader = torch.utils.data.DataLoader(
        HistoneDataset(train_genes, histone_mods=histone_mods, bin_value_type=bin_value_type, bin_size=bin_size,
                       left_flank_size=flank_size, right_flank_size=flank_size), shuffle=True,
        batch_size=n_genes_train)

    # Load test data
    test_dataloader = torch.utils.data.DataLoader(
        HistoneDataset(test_genes, histone_mods=histone_mods, bin_value_type=bin_value_type, bin_size=bin_size,
                       left_flank_size=flank_size, right_flank_size=flank_size), shuffle=False,
        batch_size=n_genes_test)

    # Run train loader
    (x_train, y_train) = next(iter(train_dataloader))
    _, n_features, n_bins = x_train.shape
    x_train = x_train.reshape(n_genes_train, n_features * n_bins)

    # Run test loader
    (x_test, y_test) = next(iter(test_dataloader))
    n_genes_test, _, _ = x_test.shape
    x_test = x_test.reshape(n_genes_test, n_features * n_bins)

    # Fit train data
    print('Fitting...')
    clf.fit(x_train, y_train)
    print('best params:', clf.best_params_)
    print(pd.DataFrame(clf.cv_results_))
    print(clf.cv_results_)

    # Predict test
    print('Predicting...')
    preds = clf.predict(x_test)
    test_score = scipy.stats.spearmanr(preds, y_test)
    preds = clf.predict(x_train)
    train_score = scipy.stats.spearmanr(preds, y_train)

    print('Spearman Correlation Score train: {}'.format(train_score))
    print('Spearman Correlation Score test: {}'.format(test_score))


def random_forest(histone_mods: list[str] = HISTONE_MODS, bin_value_type: str = 'mean', bin_size: int = 100,
                  flank_size: int = 1000):
    print(f'Random Forest, flank {flank_size}, bin {bin_size}, {bin_value_type} bin values, histones: {histone_mods}')
    rf_params = {
        'max_depth': [30],
        'n_estimators': [300],
        'bootstrap': [True],
    }
    train_and_predict(RandomForestRegressor(n_jobs=-1, random_state=42), rf_params, histone_mods, bin_value_type,
                      bin_size, flank_size)
    # All histones: Spearman correlation score 0.7511


def support_vector_machine(histone_mods: list[str] = HISTONE_MODS, bin_value_type: str = 'mean'):
    print(f'SVM, {bin_value_type} bin values, histones:{histone_mods}')
    svm_params = {
        'kernel': ['rbf'],
        'C': [10]}
    train_and_predict(svm.SVR(), svm_params, histone_mods, bin_value_type)
    # All histones: Spearman correlation score 0.7328


def analyse_histone_contribution():
    """
    RF performance
    DNase:      0.6885
    H3K4me1:    0.5321
    H3K4me3:    0.7126
    H3K9me3:    0.3205
    H3K27ac:    0.6910
    H3K27me3:   0.4446
    H3K36me3:   0.3826

    ALL: 0.7511
    without H3K9me3: 0.7498
    without H3K9me3 and H3K36me3: 0.7372

    ALL mean:   0.7511
    ALL max:    0.7484
    ALL min:    0.7511
    ALL cov.:   0.0077
    ALL std:    0.7481
    """
    for histone in HISTONE_MODS:
        random_forest(histone_mods=[histone])


def analyse_window_and_bin():
    """
    RESULTS:
    bin     flank   score
    20      1000    0.7474
    50      1000    0.7510
    100     1000    0.7484
    200     1000    0.7474
    250     1000    0.7461
    500     1000    0.7424
    1000    1000    0.7296
    2000    1000    0.7200

    200     1000    0.7476
    200     2000    0.7372
    200     5000    0.7341
    200     10000   0.7351
    200     20000
    """
    params = {
        'window_size': [5000],
        'bin_size': [200],
    }
    for flank in params['window_size']:
        for bin_size in params['bin_size']:
            random_forest(flank_size=flank, bin_size=bin_size)


if __name__ == '__main__':
    set_seed()
    random_forest(flank_size=1000, bin_size=100)
    # analyse_window_and_bin()
