import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import scipy
from scipy import stats
from tqdm import tqdm
import copy

from data_loader import *
from dataset import HistoneDataset
from histone_loader import*
from stratification import *
from modified_DeepHistone_model import DeepHistone
from modified_DeepHistone_utils import model_train,model_eval,model_predict


# Get genes (annotation and gene expression data)
train_genes, valid_genes = chromosome_splits(test_size=0.2)
n_genes_train, _ = np.shape(train_genes)
n_genes_valid, _ = np.shape(valid_genes)


# Load train data
train_dataloader = torch.utils.data.DataLoader(
    HistoneDataset(train_genes), shuffle=True, batch_size=n_genes_train)

# Load valid data
valid_dataloader = torch.utils.data.DataLoader(
    HistoneDataset(valid_genes), shuffle=False, batch_size=n_genes_valid)


# Run train loader
(x_train, y_train,genename_train) = next(iter(train_dataloader))
_, n_features, n_bins = x_train.shape
x_train = x_train.reshape(n_genes_train,1, n_features, n_bins)
y_train = y_train.reshape(n_genes_train,1,1)

# Run valid loader
(x_valid, y_valid,genename_valid) = next(iter(valid_dataloader))
n_genes_valid, _, _ = x_valid.shape
x_valid = x_valid.reshape(n_genes_valid, 1,n_features , n_bins)
y_valid = y_valid.reshape(n_genes_valid,1,1)



# Obtain data format required by DeepHistone model
train_index=list(genename_train)
valid_index=list(genename_valid)

histone_dict={genename_train[i]:x_train[i,:,:] for i in range(x_train.shape[0])}
histone_dict.update({genename_valid[i]:x_valid[i,:,:] for i in range(x_valid.shape[0])})

gex_dict={genename_train[i]:y_train[i] for i in range(x_train.shape[0])}
gex_dict.update({genename_valid[i]:y_valid[i] for i in range(x_valid.shape[0])})


# set DNA seq Input as None, updated when seq input is already
dna_dict= None #dict()



# set output folder 
model_save_file = '../data/DeepHistone/model.txt'
lab_save_file ='../data/DeepHistone/label.txt'
pred_save_file ='../data/DeepHistone/pred.txt'



# train the model 
batchsize=30
use_gpu = torch.cuda.is_available()
print('Begin training model...')
model = DeepHistone(use_gpu)
best_model = copy.deepcopy(model)
best_valid_spearmanr=0
best_valid_loss = float('Inf')

for epoch in tqdm(range(50)):
	np.random.shuffle(train_index)
	train_loss= model_train(train_index,model,batchsize,dna_dict,histone_dict,gex_dict,)
	valid_loss,valid_gex,valid_pred= model_eval(valid_index, model,batchsize,dna_dict,histone_dict,gex_dict,)
	valid_spearmanr= scipy.stats.spearmanr(valid_pred , valid_gex ).correlation

	if valid_spearmanr >best_valid_spearmanr:
		best_model = copy.deepcopy(model)

	if valid_loss < best_valid_loss: 
		early_stop_time = 0
		best_valid_loss = valid_loss	
	else:
		model.updateLR(0.1)
		early_stop_time += 1
		if early_stop_time >= 5: break

print(f"early_stop_time:{early_stop_time}")


('Begin predicting...')
valid_gex,valid_pred = model_predict(valid_index,best_model,batchsize,dna_dict,histone_dict,gex_dict,)	
valid_score = scipy.stats.spearmanr(valid_pred , valid_gex ).correlation

print('Spearman Correlation Score: {}'.format(valid_score))


print('Begin saving...')
np.savetxt(lab_save_file, valid_gex, fmt='%d', delimiter='\t')
np.savetxt(pred_save_file, valid_pred, fmt='%.4f', delimiter='\t')
best_model.save_model(model_save_file)

print('Finished.')


# check model parameters if you want
# print(model.forward_fn.parameters)