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


from modified_DeepHistone_model import *

class ModuleDense_opt1(nn.Module):
	def __init__(self,SeqOrDnase='seq',bins=None):
		super(ModuleDense_opt1, self).__init__()
		self.bins=bins
		self.tran_ksize=16
		#print("self.bins",self.bins)
		self.SeqOrDnase = SeqOrDnase
		if self.SeqOrDnase== 'seq':
			self.conv1 = nn.Sequential(
			nn.Conv2d(1,128,(4,9),1,(0,4)),
			#nn.Dropout2d(0.2),
			)
		elif self.SeqOrDnase =='dnase'  :
			self.conv1 = nn.Sequential(
			nn.Conv2d(1,128,(7,9),1,(0,4)), # here 7 is number of histone marks
			#nn.Dropout2d(0.2),
			)	
		self.block1 = DenseBlock(3, 128, 128)	
		self.trans1 = nn.Sequential(
			nn.BatchNorm2d(128+3*128),
			nn.ReLU(),
			nn.Conv2d(128+3*128, 256, (1,1),1),
			#nn.Dropout2d(0.2),
			nn.MaxPool2d((1,self.tran_ksize)),
		)
		#self.out_size = 1000 // 4 // 4  * 512 here need to be changed 
		#print(f"self.bins:{self.bins}")
		self.out_size = self.bins // self.tran_ksize  * 256 # here after eaach trans ,dimention reduce 4 times 
		#print(f"self.out_size:{self.out_size}")

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

		n, c, h, w = out.size()
		#print(self.SeqOrDnase+".out.size:",n,c,h,w)
		out = out.view(n,c*h*w) 
		return out

class NetDeepHistone_opt1(nn.Module):
	""" Here concatenate histomark Module and DNA-seq module 

	:param nn: _description_
	:type nn: _type_
	"""
	def __init__(self, bin_list=[2000,20]):
		super(NetDeepHistone_opt1, self).__init__()
		#print('DeepHistone(Dense,Dense) is used.')
		self.seq_bins,self.histone_bins=bin_list
		self.seq_map = ModuleDense_opt1(SeqOrDnase='seq',bins=self.seq_bins)
		self.seq_len = self.seq_map.out_size
		self.dns_map = ModuleDense_opt1(SeqOrDnase='dnase',bins=self.histone_bins)
		self.dns_len = self.dns_map.out_size	
		combined_len = self.dns_len  + self.seq_len 
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
		flat_seq = self.seq_map(seq)	
		#print("flat_seq.size:",flat_seq.size())

		n, h, w = dns.size()
		#print("dns.size:",n,h,w)
		dns = self.dns_map(dns) 
		flat_dns = dns.view(n,-1)
		#print("flat_dns.size",flat_dns.size())
		combined =torch.cat([flat_seq, flat_dns], 1)
		#print("combined.size:",combined.size())
		out = self.linear_map(combined)
		return out

class DeepHistone_opt1(DeepHistone):
	"""
	Build CNN model as in DeepHistone Paper. For now will just omit DNA-seq input module 

	"""

	def __init__(self,use_gpu,learning_rate=0.001,bin_list=[2000,20]):
		"""_summary_

		:param use_gpu: _description_
		:type use_gpu: _type_
		:param learning_rate: _description_, defaults to 0.001
		:type learning_rate: float, optional
		:param bin_list: number of bins (not bin size)  for DNA seq data and histone marks, defaults to [2000,20]
		:type bin_list: list, optional
		"""
		self.forward_fn = NetDeepHistone_opt1(bin_list=bin_list)  # here get general model
		self.criterion  = nn.MSELoss() #nn.BCELoss() # change loss function suit for regression model 
		self.optimizer  = optim.Adam(self.forward_fn.parameters(), lr=learning_rate, weight_decay = 0)
		self.use_gpu    = use_gpu
		if self.use_gpu : self.criterion,self.forward_fn = self.criterion.cuda(), self.forward_fn.cuda()
