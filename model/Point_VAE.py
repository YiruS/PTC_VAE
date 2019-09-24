import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable



# class PointEncoder(nn.Module):
# 	def __init__(self, zdim, input_dim=3):
# 		super(PointEncoder, self).__init__()
# 		self.zdim = zdim
# 		self.conv1 = torch.nn.Conv1d(input_dim, 128, 1)  # 64
# 		self.conv2 = torch.nn.Conv1d(128, 128, 1)  # 64
# 		self.conv3 = torch.nn.Conv1d(128, 256, 1)  # 128
# 		self.conv4 = torch.nn.Conv1d(256, 512, 1)  # 512
#
# 		self.fc1 = torch.nn.Linear(512, 256)  # 128
# 		self.fc2 = torch.nn.Linear(256, 128)  #
# 		self.fc3 = torch.nn.Linear(128, zdim)  #
#
# 		self.fcm = torch.nn.Linear(512, zdim)
# 		self.fcv = torch.nn.Linear(512, zdim)
#
# 		self.relu = nn.ReLU()
# 		self.leakyrelu = nn.LeakyReLU(0.2)
#
# 	def	forward(self, x):
# 		x = x.transpose(1,2) # BxCxN
# 		x = self.leakyrelu(self.conv1(x))
# 		x = self.leakyrelu(self.conv2(x))
# 		x = self.leakyrelu(self.conv3(x))
# 		x = self.leakyrelu(self.conv4(x))
#
# 		x = torch.max(x, 2, keepdim=True)[0]
# 		x = x.view(-1, 512)
#
# 		mu = self.fcm(x)
# 		logvar = self.fcv(x)
#
# 		# mu = self.leakyrelu(self.fc1(x))
# 		# mu = self.leakyrelu(self.fc2(mu))
# 		# mu = self.fc3(mu)
# 		#
# 		# logvar = self.leakyrelu(self.fc1(x))
# 		# logvar = self.leakyrelu(self.fc2(logvar))
# 		# logvar = self.fc3(logvar)
#
# 		return mu, logvar
#
# class PointDecoder(nn.Module):
# 	def __init__(self, zdim, out_pts=1024):
# 		super(PointDecoder, self).__init__()
# 		self.zdim = zdim
# 		self.out_pts = out_pts
# 		self.fc1 = torch.nn.Linear(zdim, 256)
# 		self.fc2 = torch.nn.Linear(256, 1024)
# 		self.fc3 = torch.nn.Linear(1024, 1024*3)
# 		self.relu = nn.ReLU()
# 		self.leakyrelu = nn.LeakyReLU(0.2)
# 		self.tanh = nn.Tanh()
#
# 	def forward(self,x):
# 		x = self.leakyrelu(self.fc1(x))
# 		x = self.leakyrelu(self.fc2(x))
# 		x = self.fc3(x)
# 		x = x.view(x.shape[0], -1, 3)
# 		x = self.tanh(x)
# 		return x

class PointVAE(nn.Module):
	def __init__(self, args, epsilon=1e-6):
		super(PointVAE, self).__init__()
		self.input_dim = args.input_dim
		self.zdim = args.zdim
		self.npts = args.npts
		self.epsilon = epsilon

		## encoder ##
		self.E_conv1 = torch.nn.Conv1d(self.input_dim, 128, 1)  # 64
		self.E_conv2 = torch.nn.Conv1d(128, 128, 1)  # 64
		self.E_conv3 = torch.nn.Conv1d(128, 256, 1)  # 128
		self.E_conv4 = torch.nn.Conv1d(256, 512, 1)  # 512

		# self.E_fc1 = torch.nn.Linear(512, 256)  # 128
		# self.E_fc2 = torch.nn.Linear(256, 128)  #
		# self.E_fc3 = torch.nn.Linear(128, self.zdim)  #

		self.E_fcm = torch.nn.Linear(512, self.zdim)
		self.E_fcv = torch.nn.Linear(512, self.zdim)

		## deconder ##
		self.D_fc1 = torch.nn.Linear(self.zdim, 256)
		self.D_fc2 = torch.nn.Linear(256, 1024)
		self.D_fc3 = torch.nn.Linear(1024, 1024 * 3)

		self.tanh = nn.Tanh()
		self.relu = nn.ReLU()
		self.leakyrelu = nn.LeakyReLU(0.2)

		# self.encoder = PointEncoder(
		# 	zdim=self.zdim,
		# 	input_dim=self.input_dim,
		# )
		# self.decoder = PointDecoder(
		# 	zdim=self.zdim,
		# 	out_pts=self.npts,
		# )

	def encoder(self, x):
		x = x.transpose(1, 2)  # BxCxN
		x = self.relu(self.E_conv1(x))
		x = self.relu(self.E_conv2(x))
		x = self.relu(self.E_conv3(x))
		x = self.relu(self.E_conv4(x))

		x = torch.max(x, 2, keepdim=True)[0]
		x = x.view(-1, 512)

		mu = self.E_fcm(x)
		sigma = self.E_fcv(x)
		stddev = self.epsilon + F.softplus(sigma)
		return mu, stddev

	def decoder(self, z):
		x = self.relu(self.D_fc1(z))
		x = self.relu(self.D_fc2(x))
		x = self.D_fc3(x)
		x = x.view(x.shape[0], -1, 3)
		x = self.tanh(x)
		return x

	def reparameterize_gaussian(self, mu, sigma):
		# std = torch.exp(0.5 * logvar)
		# eps = torch.cuda.FloatTensor(std.size()).normal_()
		# eps = Variable(eps)
		# # eps = torch.rand_like(std, device=logvar.device)
		# # eps = Variable(torch.randn(std.size(), dtype=torch.float, device=std.device))
		# return eps.mul(std).add_(mu)  # mu + std * eps
		z = mu + sigma * torch.randn_like(mu, device=sigma.device)
		z = Variable(z)
		return z

	def forward(self, x):
		z_mu, z_sigma = self.encoder(x)
		z = self.reparameterize_gaussian(z_mu, z_sigma)
		ptc = self.decoder(z)
		return ptc, z_mu, z_sigma
