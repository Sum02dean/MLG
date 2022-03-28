import numpy as np
import torch
import scipy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from data_loader import *
from dataset import *
from histone_loader import*
from stratification import *
import argparse

<<<<<<< HEAD
=======

<<<<<<< HEAD
# Parse commands
parser = argparse.ArgumentParser()

parser.add_argument("--model_name", "-mn",  type=str, default='model_x',
                    help="Name of the model")

parser.add_argument("--window_size", "-ws",  type=int, default=1000,
                    help="Number of nucleotides flanking TSS start to look at (including TSS_start)")

parser.add_argument("--bin_size", "-bs",  type=int, default=1000,
                    help="length of sequence to average histone modification values over")

# Args
args = parser.parse_args()
model_name = str(args.model_name)
window_size = int(args.window_size)
bin_size = int(args.bin_size)

=======
>>>>>>> origin/main
>>>>>>> 34-rf-baseline
# Run Script
params = {
    'max_depth': [20],
    'bootstrap': [True],
}

# Load models
model = RandomForestRegressor(n_estimators=20)
clf = GridSearchCV(estimator=model, param_grid=params, cv=3, n_jobs=-1)

# Get genes
train_genes, test_genes = chromosome_splits()
n_genes_train, _ = np.shape(train_genes)
n_genes_test, _ = np.shape(test_genes)

# Load train data
train_dataloader = torch.utils.data.DataLoader(
    HistoneDataset(train_genes,left_flank_size=window_size, right_flank_size=window_size, bin_size=bin_size), 
    shuffle=True, 
    batch_size=n_genes_train)

# Load test data
test_dataloader = torch.utils.data.DataLoader(
    HistoneDataset(train_genes,left_flank_size=window_size, right_flank_size=window_size, bin_size=bin_size),
    shuffle=False, 
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

# Store the model performance
with open (os.path.join('../outputs', '{}_spearmans.txt'.format(model_name))) as f:
    f.write(str(test_score))

print('Spearman Correlation Score: {}'.format(test_score))
print('Finished on self defined tests without errors.')

