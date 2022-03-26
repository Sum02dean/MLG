from sklearn.metrics import auc,roc_auc_score,precision_recall_curve
import numpy as np

def loadRegions(regions_indexs,dna_dict,dns_dict,label_dict,):
	if dna_dict is not None:
		dna_regions = np.concatenate([dna_dict[meta]  for meta in regions_indexs],axis=0)
	else: dna_regions =[]
	if dns_dict is not None:
		dns_regions = np.concatenate([dns_dict[meta] for meta in regions_indexs],axis=0)
	else: dns_regions =[]
	label_regions = np.concatenate([label_dict[meta] for meta in regions_indexs],axis=0).astype(int)
	return dna_regions,dns_regions,label_regions
 	
def model_train(regions,model,batchsize,dna_dict,dns_dict,label_dict,):
	train_loss = []
	regions_len = len(regions)
	for i in range(0, regions_len , batchsize):
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


def ROC(label,pred):
	if len(np.unique(np.array(label).reshape(-1)))  == 1:
		print("all the labels are the same !")
		return 0
	else:
		label = np.array(label).reshape(-1)
		pred = np.array(pred).reshape(-1)
		return roc_auc_score(label,pred)
def auPR(label,pred):
	if len(np.unique(np.array(label).reshape(-1)))  == 1:
		print("all the labels are the same !")
		return 0
	else:
		label = np.array(label).reshape(-1)
		pred = np.array(pred).reshape(-1)
		precision, recall, thresholds = precision_recall_curve(label,pred)
		return auc(recall,precision)
