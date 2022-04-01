from sklearn.metrics import auc,roc_auc_score,precision_recall_curve
import numpy as np


def get_reshaped_data(dataloader):
	"""Reshape data to fit into DeepHistone Model 

	:param dataloader: HistoneDataset
	:type dataloader: HistoneDataset wraped by torch.utils.data.DataLoader
	"""
	(x, y,genename) = next(iter(dataloader)) # this step is slow since it actually loads all data 
	x_histone,x_seq=x
	#print(x_histone.shape,x_seq.shape,y.shape,len(genename),len(set(genename)))

	n_genes, n_features_histone, n_bins_histone = x_histone.shape
	x_histone = x_histone.reshape(n_genes,1, n_features_histone, n_bins_histone)

	_, n_bins_seq,n_features_seq = x_seq.shape
	x_seq = x_seq.reshape(n_genes,1, n_features_seq, n_bins_seq)

	y = y.reshape(n_genes,1,1)
	#print(x_histone.shape,x_seq.shape,y.shape)

	return(x_histone,x_seq,y,list(genename))


def get_dict_from_data(train_index,valid_index,test_index,train,valid,test):
	"""_summary_

	:param train_index: _description_
	:type train_index: _type_
	:param valid_index: _description_
	:type valid_index: _type_
	:param test_index: _description_
	:type test_index: _type_
	:param train: _description_
	:type train: _type_
	:param valid: _description_
	:type valid: _type_
	:param test: _description_
	:type test: _type_
	"""
	return_dict= {train_index[i]:train[i,...] for i in range(train.shape[0])}
	#print(len(return_dict))

	return_dict.update({valid_index[i]:valid[i,...] for i in range(valid.shape[0])})
	#print(len(return_dict))

	return_dict.update({test_index[i]:test[i,...] for i in range(test.shape[0])})
	#print(len(return_dict))

	return(return_dict)



def loadRegions(regions_indexs,dna_dict,dns_dict,label_dict,):
	if dna_dict is not None:
		dna_regions = np.concatenate([dna_dict[meta]  for meta in regions_indexs],axis=0)
	else: dna_regions =[]
	if dns_dict is not None:
		dns_regions = np.concatenate([dns_dict[meta] for meta in regions_indexs],axis=0)
	else: dns_regions =[]
	label_regions = np.concatenate([label_dict[meta] for meta in regions_indexs],axis=0) #.astype(int) ; here our output is regression value 
	return dna_regions,dns_regions,label_regions
 	
def model_train(regions,model,batchsize,dna_dict,dns_dict,label_dict,):
	train_loss = []
	regions_len = len(regions)
	for i in range(0, regions_len , batchsize):
		#for testing reason add this 
		if i % 100 ==0:
			print(f"batch_idx: {i}")
		regions_batch = [regions[i+j] for j in range(batchsize) if (i+j) < regions_len]
		#print("region_batch: ",(regions_batch))
		seq_batch ,dns_batch,lab_batch = loadRegions(regions_batch,dna_dict,dns_dict,label_dict)
		_loss= model.train_on_batch(seq_batch, dns_batch, lab_batch)
		train_loss.append(_loss)
	return np.mean(train_loss) 

def model_eval(regions,model,batchsize,dna_dict,dns_dict,label_dict,):
	loss = []
	pred =[]
	lab =[]
	regions_len = len(regions)
	for i in range(0, regions_len , batchsize):
		regions_batch = [regions[i+j] for j in range(batchsize) if (i+j) < regions_len]
		seq_batch ,dns_batch,lab_batch = loadRegions(regions_batch,dna_dict,dns_dict,label_dict)
		_loss,_pred = model.eval_on_batch(seq_batch, dns_batch, lab_batch)
		loss.append(_loss)
		lab.extend(lab_batch)
		pred.extend(_pred)
	return np.mean(loss), np.array(lab),np.array(pred)

def model_predict(regions,model,batchsize,dna_dict,dns_dict,label_dict,):
	lab  = []
	pred = []
	regions_len = len(regions)
	for i in range(0, len(regions), batchsize):
		regions_batch = [regions[i+j] for j in range(batchsize) if (i+j) < regions_len]
		seq_batch ,dns_batch,lab_batch = loadRegions(regions_batch,dna_dict,dns_dict,label_dict)
		_pred = model.test_on_batch(seq_batch, dns_batch)
		lab.extend(lab_batch)
		pred.extend(_pred)		
	return np.array(lab), np.array(pred) 

