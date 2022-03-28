import numpy as np
import torch
import scipy
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from dataset import *
from histone_loader import*
from stratification import *

# Run Script
params = {
    'kernel': ['rbf'],
     'C':[10]}

# Load models
model = svm.SVR()
clf = GridSearchCV(estimator=model, param_grid=params, cv=3, n_jobs=-1)

# Get genes
train_genes, test_genes = chromosome_splits()
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

# Fit train data
clf.fit(x_train, y_train)

# Predict test
preds = clf.predict(x_test)
test_score = scipy.stats.spearmanr(preds, y_test)

print('Spearman Correlation Score: {}'.format(test_score))
print('Finished on self defined tests without errors.')
