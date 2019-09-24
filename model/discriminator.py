
'''
GAN used in the paper[1]:
Reference:
[1] Bousmalis, Konstantinos, et al.
	"Unsupervised Pixel-level Domain Adaptation with GANs." (2017)..
'''

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

############################
# Discrminator
############################

class PointNetDiscriminator(nn.Module):
	def __init__(self, k=9, input_dim=3, out_dim=1):
		super(PointNetDiscriminator, self).__init__()
		self.conv1 = torch.nn.Conv1d(input_dim, 64, 1) # 64
		self.conv2 = torch.nn.Conv1d(64, 64, 1) # 64
		self.conv3 = torch.nn.Conv1d(64, 128, 1) # 128
		self.conv4 = torch.nn.Conv1d(128, 512, 1)  # 512

		self.fc1 = torch.nn.Linear(512, 128) # 128
		self.fc2 = torch.nn.Linear(128, 64) #
		self.fc3 = torch.nn.Linear(64, out_dim)  #

		self.cls = torch.nn.Linear(64, k)

		self.dropout = nn.Dropout(p=0.5)

		self.leaky = nn.LeakyReLU(0.2, inplace=False)

	def forward(self, x):
		x = x.transpose(2,1) # BxCxN
		x = self.leaky(self.conv1(x))
		x = self.leaky(self.conv2(x))
		x = self.leaky(self.conv3(x))
		x = self.conv4(x)

		x = x.transpose(2,1) # BxNxC
		x = torch.max(x, 1, keepdim=True)[0] #Bx1xC

		x = self.leaky(self.fc1(x))
		x = self.leaky(self.fc2(x))
		out_cls = self.cls(x)
		x = self.fc3(x)

		return x, out_cls.view(x.shape[0],-1)


