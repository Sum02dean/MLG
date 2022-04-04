import pandas as pd
import numpy as np
import torch

import scipy
from scipy import stats
from tqdm import tqdm
import copy
import time
import logging




from utils.data_loader import *
from utils.dataset import HistoneDataset_returngenenames

from modified_DeepHistone_model import DeepHistone
from modified_DeepHistone_utils import model_train,model_eval,model_predict
from modified_DeepHistone_utils import get_reshaped_data
from modified_DeepHistone_utils import get_dict_from_data
from modified_DeepHistone_utils import get_dict_from_data_submisson
from modified_DeepHistone_utils import create_submission



# model_save_folder="../data/DeepHistone/"


# left_flank_size = 1000#500#1000
# right_flank_size = left_flank_size#500#1000
# seq_bin_size=left_flank_size+right_flank_size
# histone_bin_size = 1 #100 ,20 ,5,1

# seq_bins=seq_bin_size
# assert seq_bin_size % histone_bin_size==0
# histone_bins=int(seq_bin_size/histone_bin_size)


# use_gpu = torch.cuda.is_available()


# batchsize=30#10000 # 20, 30

# conv_ksize,tran_ksize=9,4 

# use_seq=False



# best_model_name="prefix-basic-model--use_seq-False-left_flank_size-1000-histone_bin_size-1-conv_ksize-9-tran_ksize-4-batchsize-30-epochs-30-use_gpu-True-time20220404-103222-epoch18-model-final.txt"



# valid_chr=[5, 10, 15, 20]
# test_chr=[2,18]
# train_chr=[i for i in range(1,23) if (i not in valid_chr+test_chr)]
# all_genes = load_train_genes()
# all_genes.head(n=3)


# # Get genes
# train_genes=filter_genes_by_chr(all_genes,train_chr)
# valid_genes=filter_genes_by_chr(all_genes,valid_chr)
# test_genes=filter_genes_by_chr(all_genes,test_chr)


# n_genes_train, _ = np.shape(train_genes)
# n_genes_valid, _ = np.shape(valid_genes)
# n_genes_test, _ = np.shape(test_genes)

# submission_genes=load_test_genes()
# submission_genes['cell_line'] = 3
# n_genes_submission, _ = np.shape(submission_genes)
# submission_genes.head(n=3)





# # Load submission data 
# submission_dataloader = torch.utils.data.DataLoader(
#     HistoneDataset_returngenenames(submission_genes,left_flank_size=left_flank_size,right_flank_size=right_flank_size,bin_size=histone_bin_size,use_seq=True), 
#     shuffle=False, batch_size=n_genes_submission)

# print("finish load data")


# # Run submission loader
# x_submission_histone,x_submission_seq,submission_index=get_reshaped_data(dataloader=submission_dataloader,is_train=False)

# print("finish run  loader")



# submission_dna_dict= get_dict_from_data_submisson(submission_index,x_submission_seq)

# submission_histone_dict= get_dict_from_data_submisson(submission_index,x_submission_histone,)


# print("finish get dictionary")


# model = DeepHistone(use_gpu,use_seq=use_seq,bin_list=[seq_bins,histone_bins],inside_ksize=[conv_ksize,tran_ksize])
# model.forward_fn.load_state_dict(torch.load(f"{model_save_folder}{best_model_name}"))


# print('Begin predicting...')
# submission_pred = model_predict(submission_index,model,batchsize,submission_dna_dict,submission_histone_dict,None,)
# print(submission_pred.shape)

# create_submission(submission_genes,submission_pred.flatten())