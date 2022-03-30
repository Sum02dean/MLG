import scipy
from sklearn import svm
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from dataset import *
from stratification import *
from utils.histone_loader import VALUE_TYPES


def train_and_predict(model: BaseEstimator, params: dict, histone_mods: list[str], bin_value_type: str):
    # Load models
    clf = GridSearchCV(estimator=model, param_grid=params, cv=3, n_jobs=-1)

    # Get genes
    train_genes, test_genes = chromosome_splits()
    n_genes_train, _ = np.shape(train_genes)
    n_genes_test, _ = np.shape(test_genes)

    # Load train data
    train_dataloader = torch.utils.data.DataLoader(
        HistoneDataset(train_genes, histone_mods=histone_mods, bin_value_type=bin_value_type), shuffle=True,
        batch_size=n_genes_train)

    # Load test data
    test_dataloader = torch.utils.data.DataLoader(
        HistoneDataset(test_genes, histone_mods=histone_mods, bin_value_type=bin_value_type), shuffle=False,
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
    clf.fit(x_train, y_train)

    # Predict test
    preds = clf.predict(x_test)
    test_score = scipy.stats.spearmanr(preds, y_test)

    print('Spearman Correlation Score: {}'.format(test_score))


def random_forest(histone_mods: list[str] = HISTONE_MODS, bin_value_type: str = 'mean'):
    print(f'Random Forest, {bin_value_type} bin values, histones:{histone_mods}')
    rf_params = {
        'max_depth': [20],
        'bootstrap': [True],
    }
    train_and_predict(RandomForestRegressor(n_estimators=20, random_state=42), rf_params, histone_mods, bin_value_type)
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


if __name__ == '__main__':
    for vt in VALUE_TYPES:
        random_forest(bin_value_type=vt)
