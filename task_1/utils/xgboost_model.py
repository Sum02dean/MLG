
import numpy as np
import xgboost as xgb
import scipy
import torch
from data_loader import *
from dataset import *
from histone_loader import*
from tqdm import tqdm
from stratification import chromosome_splits
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, Normalizer


SCALE = False
SAVE_DATA = True

def spearman_scoring(y, y_true):
    return scipy.stats.spearmanr(y, y_true)[0]

def to_pandas(x):
    return pd.DataFrame(x.numpy())

# Get genes
train_genes, _ = chromosome_splits(cell_line=1, test_size=0.1)
_, test_genes = chromosome_splits(cell_line=2, test_size=0.1)
n_genes_train, _ = np.shape(train_genes)
n_genes_test, _ = np.shape(test_genes)


# Load train data
train_dataloader = torch.utils.data.DataLoader(
    HistoneDataset(train_genes), shuffle=True, batch_size=n_genes_train)

# Load test data
test_dataloader = torch.utils.data.DataLoader(
    HistoneDataset(test_genes), shuffle=False, batch_size=n_genes_test)

# Run train loader
(x_train, y_train) = next(iter(train_dataloader))
_, n_features, n_bins = x_train.shape
x_train = x_train.reshape(n_genes_train, n_features * n_bins)

# Run test loader
(x_test, y_test) = next(iter(test_dataloader))
n_genes_test, _, _ = x_test.shape
x_test = x_test.reshape(n_genes_test, n_features * n_bins)

# Save csv
if SAVE_DATA:
    to_pandas(x_train).to_csv('x_train.csv')
    to_pandas(y_train).to_csv('y_train.csv')
    to_pandas(x_test).to_csv('x_test.csv')
    to_pandas(y_test).to_csv('y_test.csv')
    print("saving complete")
# Scale features
if SCALE:
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)

# Construct params dict
params = {'max_depth': [15],
          'eta': [0.1],
          'alpha': [0.1],
          'lambda': [0.01],
          'subsample': [0.9],
          'colsample_bynode': [0.2]}

# Construct model + classifiers
model = xgb.XGBRegressor()

# Spearmans score
custom_scorer = make_scorer(
    spearman_scoring, greater_is_better=True)

# Construct clf
clf = GridSearchCV(
    model, params, n_jobs=-1, cv=5, 
    scoring=custom_scorer
    )


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

# Report
print('Spearman Correlation Score train: {}'.format(train_score))
print('Spearman Correlation Score test: {}'.format(test_score))


# 0.5969965823754 on inter cell-line splits












