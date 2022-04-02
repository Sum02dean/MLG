from typing import final
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import scipy
from scipy import stats
from tqdm import tqdm
import copy

from data_loader import *
from dataset import HistoneDataset_returngenenames
from histone_loader import*
from stratification import *
from modified_DeepHistone_model import DeepHistone
from modified_DeepHistone_utils import model_train,model_eval,model_predict
from modified_DeepHistone_utils import get_reshaped_data
from modified_DeepHistone_utils import get_dict_from_data
from modified_DeepHistone_utils import save_model


model_save_folder="../data/DeepHistone/"


# Get genes, notive here test_genes not refer to final test dataset used for submission
# but subset from whole training dataset 
valid_chr=[5,20]
test_chr=[2]
train_chr=[i for i in range(1,23) if (i not in valid_chr+test_chr)]
print(train_chr,valid_chr,test_chr)

all_genes = load_train_genes()
train_genes=filter_genes_by_chr(all_genes,train_chr)
valid_genes=filter_genes_by_chr(all_genes,valid_chr)
test_genes=filter_genes_by_chr(all_genes,test_chr)

n_genes_train, _ = np.shape(train_genes)
n_genes_valid, _ = np.shape(valid_genes)
n_genes_test, _ = np.shape(test_genes)
print(train_genes.shape,valid_genes.shape,test_genes.shape)

# set some hyper-parameters
left_flank_size = 500#1000
right_flank_size = 500#1000
seq_bin_size=left_flank_size+right_flank_size
histone_bin_size = 20 #100

seq_bins=seq_bin_size
assert seq_bin_size % histone_bin_size==0
histone_bins=int(seq_bin_size/histone_bin_size)
print(seq_bins,histone_bins)

# Load train data
train_dataloader = torch.utils.data.DataLoader(
    HistoneDataset_returngenenames(train_genes,left_flank_size=left_flank_size,right_flank_size=right_flank_size,bin_size=histone_bin_size,use_seq=True), 
    shuffle=False, batch_size=n_genes_train)

# Load valid data
valid_dataloader = torch.utils.data.DataLoader(
    HistoneDataset_returngenenames(valid_genes,left_flank_size=left_flank_size,right_flank_size=right_flank_size,bin_size=histone_bin_size,use_seq=True), 
    shuffle=False, batch_size=n_genes_valid)

# Load test data
test_dataloader = torch.utils.data.DataLoader(
    HistoneDataset_returngenenames(test_genes,left_flank_size=left_flank_size,right_flank_size=right_flank_size,bin_size=histone_bin_size,use_seq=True), 
    shuffle=False, batch_size=n_genes_valid)


# get DeepHistone required data format 
x_train_histone,x_train_seq,y_train,train_index=get_reshaped_data(dataloader=train_dataloader)
x_valid_histone,x_valid_seq,y_valid,valid_index=get_reshaped_data(dataloader=valid_dataloader)
x_test_histone,x_test_seq,y_test,test_index=get_reshaped_data(dataloader=test_dataloader)
print(len(train_index),len(valid_index),len(test_index))

dna_dict= get_dict_from_data(train_index,valid_index,test_index,
                             x_train_seq,x_valid_seq,x_test_seq)
histone_dict= get_dict_from_data(train_index,valid_index,test_index,
                             x_train_histone,x_valid_histone,x_test_histone)
gex_dict = get_dict_from_data(train_index,valid_index,test_index,
                             y_train,y_valid,y_test)




use_gpu = torch.cuda.is_available()
batchsize=30#10000 # 20, 30
epochs=3 #10 #50

print('Begin training model...')
model = DeepHistone(use_gpu,bin_list=[seq_bins,histone_bins])
best_model = copy.deepcopy(model)
best_valid_spearmanr=0
best_valid_loss = float('Inf')

for epoch in tqdm(range(epochs)):
	np.random.shuffle(train_index)
	train_loss= model_train(train_index,model,batchsize,dna_dict,histone_dict,gex_dict,)
	valid_loss,valid_gex,valid_pred= model_eval(valid_index, model,batchsize,dna_dict,histone_dict,gex_dict,)
	valid_spearmanr= scipy.stats.spearmanr(valid_pred , valid_gex ).correlation
	print(f"epoch:{epoch} valid_loss:{valid_loss} valid_spearmanr:{valid_spearmanr}")
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



('Begin predicting use besting model...')
test_gex,test_pred = model_predict(test_index,best_model,batchsize,dna_dict,histone_dict,gex_dict,)	
test_score = scipy.stats.spearmanr(test_pred , test_gex ).correlation
print('Spearman Correlation Score: {}'.format(test_score))


('Begin predicting use final model...')
test_gex,test_pred = model_predict(test_index,model,batchsize,dna_dict,histone_dict,gex_dict,)	
test_score = scipy.stats.spearmanr(test_pred , test_gex ).correlation

print('Spearman Correlation Score: {}'.format(test_score))



print('Begin saving...')
np.savetxt(f"{model_save_folder}label.txt", valid_gex, fmt='%d', delimiter='\t')
np.savetxt(f"{model_save_folder}pred.txt", valid_pred, fmt='%.4f', delimiter='\t')
save_model(model=best_model,epoch=epoch,seq_bins=seq_bins,histone_bins=histone_bins,
            model_save_folder=model_save_folder,prefix="",suffix="best")
save_model(model=best_model,epoch=epoch,seq_bins=seq_bins,histone_bins=histone_bins,
            model_save_folder=model_save_folder,prefix="",suffix="final")



print('Finished.')




# check model parameters if you want
# print(model.forward_fn.parameters)