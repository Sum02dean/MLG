import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn import metrics
from torch.optim import  Optimizer
import math 
from torch.nn.parameter import Parameter

class BasicBlock(nn.Module):
	def __init__(self, in_planes, grow_rate,conv_ksize,conv_padsize):
		super(BasicBlock, self).__init__()
		self.block = nn.Sequential(
			nn.BatchNorm2d(in_planes),
			nn.ReLU(),
			nn.Conv2d(in_planes, grow_rate, (1,conv_ksize), 1, (0,conv_padsize)), # here 9 and 4 are choosed to keep raw dimention intact 
			#nn.Dropout2d(0.2)
		)
	def forward(self, x):
		out = self.block(x)
		return torch.cat([x, out],1)

class DenseBlock(nn.Module):
	def __init__(self, nb_layers, in_planes, grow_rate,conv_ksize,conv_padsize):
		super(DenseBlock, self).__init__()
		layers = []
		for i in range(nb_layers):
			layers.append(BasicBlock(in_planes + i*grow_rate, grow_rate,conv_ksize,conv_padsize,))
		self.layer = nn.Sequential(*layers)
	def forward(self, x):
		return self.layer(x)


class ModuleDense(nn.Module):
	def __init__(self,SeqOrDnase='seq',bins=None,inside_ksize=[9,4]):
		super(ModuleDense, self).__init__()
		self.bins=bins
		self.conv_ksize,self.tran_ksize=inside_ksize
		self.conv_padsize=self.conv_ksize//2
		self.SeqOrDnase = SeqOrDnase
		if self.SeqOrDnase== 'seq':
			self.conv1 = nn.Sequential(
			nn.Conv2d(1,128,(4,self.conv_ksize),1,(0,self.conv_padsize)),
			#nn.Dropout2d(0.2),
			)
		elif self.SeqOrDnase =='dnase'  :
			self.conv1 = nn.Sequential(
			nn.Conv2d(1,128,(7,self.conv_ksize),1,(0,self.conv_padsize)), # here 7 is number of histone marks
			#nn.Dropout2d(0.2),
			)	
		self.block1 = DenseBlock(3, 128, 128,self.conv_ksize,self.conv_padsize)	
		self.trans1 = nn.Sequential(
			nn.BatchNorm2d(128+3*128),
			nn.ReLU(),
			nn.Conv2d(128+3*128, 256, (1,1),1),
			#nn.Dropout2d(0.2),
			nn.MaxPool2d((1,self.tran_ksize)),
		)
		self.block2 = DenseBlock(3,256,256,self.conv_ksize,self.conv_padsize)
		self.trans2 = nn.Sequential(
			nn.BatchNorm2d(256+3*256),
			nn.ReLU(),
			nn.Conv2d(256+3*256, 512, (1,1),1),
			#nn.Dropout2d(0.2),
			nn.MaxPool2d((1,self.tran_ksize)), # default stride id 4 as kernel size
		)
		#self.out_size = 1000 // 4 // 4  * 512 here need to be changed 
		#print(f"self.bins:{self.bins}")
		self.out_size = self.bins // self.tran_ksize // self.tran_ksize  * 512

	def forward(self, seq):
		n, h, w = seq.size()
		#print("seq.size(),n,h,w",n,h,w)
		if self.SeqOrDnase=='seq':
			seq = seq.view(n,1,4,w)
		elif self.SeqOrDnase=='dnase':
			seq = seq.view(n,1,7,w) # seq = seq.view(n,1,1,w)  here w=20 .bin size
		#print("self.SeqOrDnase:",self.SeqOrDnase,";self.seq.size",seq.size())
		out = self.conv1(seq)
		#print("self.conv1.size",out.size())
		out = self.block1(out)
		#print("self.block1.size",out.size())
		out = self.trans1(out)
		#print("self.trans1.size",out.size())
		out = self.block2(out)
		#print("self.block2.size",out.size())
		out = self.trans2(out)
		#print("self.trans2.size",out.size())
		n, c, h, w = out.size()
		#print(self.SeqOrDnase+".out.size:",n,c,h,w)
		out = out.view(n,c*h*w) 
		return out



class NetDeepHistone(nn.Module):
	""" Here concatenate histomark Module and DNA-seq module 

	:param nn: _description_
	:type nn: _type_
	"""
	def __init__(self,use_seq,bin_list,inside_ksize):
		super(NetDeepHistone, self).__init__()
		#print('DeepHistone(Dense,Dense) is used.')
		self.use_seq=use_seq
		self.seq_bins,self.histone_bins=bin_list

		self.dns_map = ModuleDense(SeqOrDnase='dnase',bins=self.histone_bins,inside_ksize=inside_ksize)
		self.dns_len = self.dns_map.out_size	

		if self.use_seq:
			self.seq_map = ModuleDense(SeqOrDnase='seq',bins=self.seq_bins,inside_ksize=inside_ksize)
			self.seq_len = self.seq_map.out_size
			combined_len = self.dns_len  + self.seq_len 
		else:
			combined_len = self.dns_len 
		#print("combined_len:", combined_len)
		self.linear_map = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(int(combined_len),925),
			nn.BatchNorm1d(925),
			nn.ReLU(),
			#nn.Dropout(0.1), # this one is not delete by me 
			nn.Linear(925,1), # nn.Linear(925,7), here we only want to predict 1 output/gene expression
			nn.ReLU(),
			#nn.Sigmoid(), as now its a regression problem not a classifction problem
		)

	def forward(self, seq, dns):
		dns_n, dns_h, dns_w = dns.size()
		#print("dns.size:",dns_n,dns_h,dns_w)
		dns = self.dns_map(dns) 
		#print("dns.size:",dns.size())
		flat_dns = dns.view(dns_n,-1) # so here actully is strang , why seq not needed flatting ?
		# okay , so this step is acutlly not neceseary 
		#print("flat_dns.size:",flat_dns.size())
		if self.use_seq:
			seq_n, seq_h, seq_w = seq.size()
			flat_seq = self.seq_map(seq)	
			#print("flat_seq.size:",flat_seq.size())
			#print("seq.size:",seq_n,seq_h,seq_w)
			combined =torch.cat([flat_seq, flat_dns], 1)
		else:
			combined=flat_dns
	
		
		#print("combined.size:",combined.size())
		out = self.linear_map(combined)
		return out


class DeepHistone():
	"""
	Build CNN model as in DeepHistone Paper. For now will just omit DNA-seq input module 

	"""

	def __init__(self,use_gpu,learning_rate=0.001,use_seq=True,bin_list=[2000,20],inside_ksize=[9,4]):
		"""_summary_

		:param use_gpu: _description_
		:type use_gpu: _type_
		:param learning_rate: _description_, defaults to 0.001
		:type learning_rate: float, optional
		:param bin_list: number of bins (not bin size)  for DNA seq data and histone marks, defaults to [2000,20]
		:type bin_list: list, optional
		:param inside_ksize: kerner size for model (conv_ksize,trans_ksize=), defaults to [9,4]
		:type inside_ksize: list, optional
		"""
		self.forward_fn = NetDeepHistone(use_seq=use_seq,bin_list=bin_list,inside_ksize=inside_ksize)  # here get general model
		self.criterion  = nn.MSELoss() #nn.BCELoss() # change loss function suit for regression model 
		self.optimizer  = optim.Adam(self.forward_fn.parameters(), lr=learning_rate, weight_decay = 0)
		self.use_gpu    = use_gpu
		if self.use_gpu : self.criterion,self.forward_fn = self.criterion.cuda(), self.forward_fn.cuda()

	def updateLR(self, fold):
		for param_group in self.optimizer.param_groups:
			param_group['lr'] *= fold

	def train_on_batch(self,seq_batch,dns_batch,lab_batch,): 
		self.forward_fn.train()
		seq_batch  = Variable(torch.Tensor(seq_batch))
		dns_batch  = Variable(torch.Tensor(dns_batch))
		lab_batch  = Variable(torch.Tensor(lab_batch))
		if self.use_gpu: seq_batch, dns_batch, lab_batch = seq_batch.cuda(), dns_batch.cuda(), lab_batch.cuda()
		output = self.forward_fn(seq_batch, dns_batch)
		loss = self.criterion(output,lab_batch)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss.cpu().data

	def eval_on_batch(self,seq_batch,dns_batch,lab_batch,):
		self.forward_fn.eval()
		seq_batch  = Variable(torch.Tensor(seq_batch))
		dns_batch  = Variable(torch.Tensor(dns_batch))
		lab_batch  = Variable(torch.Tensor(lab_batch))
		if self.use_gpu: seq_batch, dns_batch, lab_batch = seq_batch.cuda(), dns_batch.cuda(), lab_batch.cuda()
		output = self.forward_fn(seq_batch, dns_batch)
		loss = self.criterion(output,lab_batch)
		return loss.cpu().data,output.cpu().data.numpy()
			
	def test_on_batch(self, seq_batch, dns_batch):
		self.forward_fn.eval()
		seq_batch  = Variable(torch.Tensor(seq_batch))
		dns_batch  = Variable(torch.Tensor(dns_batch))
		if self.use_gpu: seq_batch, dns_batch,  = seq_batch.cuda(), dns_batch.cuda()
		output = self.forward_fn(seq_batch, dns_batch)
		pred = output.cpu().data.numpy()
		return pred
	
	def save_model(self, path):
		torch.save(self.forward_fn.state_dict(), path)


	def load_model(self, path):
		self.forward_fn.load_state_dict(torch.load(path))



