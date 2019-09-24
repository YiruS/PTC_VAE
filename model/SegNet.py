from __future__ import absolute_import, division, print_function, unicode_literals

import cv2
import optparse
import sys
import os
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/.." % file_path)
from utils.torchsummary import summary

def load_pretrained_vgg(model, vgg_model):
	'''
	module to load pretrained_vgg model weights
	'''
	count_conv=0
	for m in model.modules():
		if isinstance(m,nn.Conv2d):
			count_conv+=1
	print (count_conv)
	count=0
	for m in model.modules():
		for v in vgg_model.modules():
			if isinstance(m,nn.Conv2d) and isinstance(v,nn.Conv2d) and count<count_conv:
				if m.weight.size()==v.weight.size():
					m.weight.data=v.weight.data
					m.bias.data=v.bias.data
		if isinstance(m,nn.Conv2d):
			count+=1
	return model


### define layers:



class _EncoderBlock(nn.Module):
	'''
	Encoder block structured aroung vgg_model blocks
	'''
	def __init__(self, in_channels, out_channels, kernel_size, separable_conv=False,name='default',BN=True):
		super(_EncoderBlock, self).__init__()
		self.name=name

		_encoder_layer_SC_WBN=[
		nn.Conv2d(in_channels,in_channels,kernel_size=kernel_size,stride=1,padding=1,groups=in_channels),
		nn.BatchNorm2d(in_channels),
		nn.ReLU(inplace=True),
		nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(inplace=True),
		]

		_encoder_layer_SC_NBN=[
		nn.Conv2d(in_channels,in_channels,kernel_size=kernel_size,stride=1,padding=1,groups=in_channels),
		nn.ReLU(inplace=True),
		nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1),
		nn.ReLU(inplace=True),
		]

		if not separable_conv:
			layers = [
				nn.Conv2d(
					in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1),
				nn.BatchNorm2d(out_channels),
				nn.ReLU(inplace=True),
			]
		else:
			if BN:
				layers=_encoder_layer_SC_WBN
			else:
				layers=_encoder_layer_SC_NBN
		self.encode = nn.Sequential(*layers)

	def forward(self, x):
		return self.encode(x)


class _DecoderBlock(nn.Module):
	'''
	Decoder blocks using Transpose Convolution blocks
	'''
	def __init__(self, in_channels, out_channels, kernel_size,is_nonlinear=True,separable_conv=False,name='default',BN=True):
		super(_DecoderBlock, self).__init__()
		self.name=name
		_decoder_layer_SC_WBN=[
		nn.Conv2d(in_channels,in_channels,kernel_size=kernel_size,stride=1,padding=1,groups=in_channels),
		nn.BatchNorm2d(in_channels),
		nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1),
		nn.BatchNorm2d(out_channels),]

		_decoder_layer_SC_NBN=[
		nn.Conv2d(in_channels,in_channels,kernel_size=kernel_size,stride=1,padding=1,groups=in_channels),
		nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1),]

		if not separable_conv:
			layers = [
				nn.ConvTranspose2d(
					in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1),
				nn.BatchNorm2d(out_channels),
			]
		else:
			if BN:
				layers=_decoder_layer_SC_WBN
			else:
				layers=_decoder_layer_SC_NBN

		if is_nonlinear:
			layers.append(nn.ReLU(inplace=True))

		self.decode = nn.Sequential(*layers)

	def forward(self, x):
		return self.decode(x)

class BR(nn.Module):
	'''
	Boundry refinement block
	See: https://arxiv.org/pdf/1703.02719.pdf
	'''
	def __init__(self, out_c):
		super(BR, self).__init__()
		# self.bn = nn.BatchNorm2d(out_c)
		self.relu = nn.ReLU(inplace=False)
		self.conv1 = nn.Conv2d(out_c,out_c, kernel_size=3,padding=1)
		self.conv2 = nn.Conv2d(out_c,out_c, kernel_size=3,padding=1)

	def forward(self,x):
		x_res = self.conv1(x)
		x_res = self.relu(x_res)
		x_res = self.conv2(x_res)

		x = x + x_res

		return x

class SegNet_Small(nn.Module):
	'''
	Low complexity version of SegNet Semantic Segmentation model
	Designed for eye feature segmentation task
	'''
	def __init__(self, input_channels, num_classes,skip_type=None,BR_bool=False,separable_conv=False,caffe=False,mode='nearest',BN=True,latent=False):
		super(SegNet_Small, self).__init__()
		self.BR_bool=BR_bool
		self.skip_type=skip_type
		self.caffe=caffe
		self.mode=mode
		self.BN=BN
		self.latent = latent
		self.enc10 = _EncoderBlock(input_channels, 64, 3,separable_conv=separable_conv,name='enc10',BN=self.BN)
		self.enc11 = _EncoderBlock(64, 64, 3,separable_conv=separable_conv,name='enc11',BN=self.BN)
		self.enc20 = _EncoderBlock(64, 128, 3,separable_conv=separable_conv,name='enc20',BN=self.BN)
		self.enc21 = _EncoderBlock(128, 128, 3,separable_conv=separable_conv,name='enc21',BN=self.BN)
		self.enc30 = _EncoderBlock(128, 256, 3,separable_conv=separable_conv,name='enc30',BN=self.BN)
		self.enc31 = _EncoderBlock(256, 256, 3,separable_conv=separable_conv,name='enc31',BN=self.BN)
		self.enc32 = _EncoderBlock(256, 256, 3,separable_conv=separable_conv,name='enc32',BN=self.BN)

		self.dec32 = _DecoderBlock(256, 256, 3,separable_conv=separable_conv,name='dec32',BN=self.BN)
		self.dec31 = _DecoderBlock(256, 256, 3,separable_conv=separable_conv,name='dec31',BN=self.BN)
		self.dec30 = _DecoderBlock(256, 128, 3,separable_conv=separable_conv,name='dec30',BN=self.BN)
		self.dec21 = _DecoderBlock(128, 128, 3,separable_conv=separable_conv,name='dec21',BN=self.BN)
		self.dec20 = _DecoderBlock(128, 64, 3,separable_conv=separable_conv,name='dec20',BN=self.BN)
		self.dec11 = _DecoderBlock(64, 64, 3,separable_conv=separable_conv,name='dec11',BN=self.BN)
		self.dec10 = _DecoderBlock(64, num_classes, 3,is_nonlinear=False,separable_conv=separable_conv,name='dec10',BN=self.BN)
		if self.BR_bool:
			self.BR=BR(num_classes)
		# initialize_weights(self.enc10,self.enc11,self.enc20,self.enc21,self.enc30,\
		# self.enc31,self.enc32,self.dec32,self.dec31,self.dec30,self.dec21,self.dec20,\
		# self.dec11,self.dec10)

	def forward(self, x):
		dim_0 = x.size()

		enc1 = self.enc10(x)
		enc1 = self.enc11(enc1)
		x_1, indices_1 = F.max_pool2d(
			enc1, kernel_size=2, stride=2, return_indices=True
		)

		dim_1 = x_1.size()
		enc2 = self.enc20(x_1)
		enc2 = self.enc21(enc2)
		x_2, indices_2 = F.max_pool2d(
			enc2, kernel_size=2, stride=2, return_indices=True
		)

		dim_2 = x_2.size()
		enc3 = self.enc30(x_2)
		enc3 = self.enc31(enc3)
		enc3 = self.enc32(enc3)
		x_3, indices_3 = F.max_pool2d(
			enc3, kernel_size=2, stride=2, return_indices=True
		)

		if self.caffe:
			dec3=nn.functional.interpolate(x_3,scale_factor=2, mode=self.mode)
		else:
			dec3 = F.max_unpool2d(
				x_3, indices_3, kernel_size=2, stride=2, output_size=dim_2
			)
		dec3 = self.dec32(dec3)
		dec3 = self.dec31(dec3)
		dec3 = self.dec30(dec3)

		if self.caffe:
			dec2=nn.functional.interpolate(dec3,scale_factor=2, mode=self.mode)
		else:
			dec2 = F.max_unpool2d(
				dec3, indices_2, kernel_size=2, stride=2, output_size=dim_1
			)
		if self.skip_type is not None:
			dec2+=enc2
		dec2 = self.dec21(dec2)
		dec2 = self.dec20(dec2)
		if self.caffe:
			dec1=nn.functional.interpolate(dec2,scale_factor=2, mode=self.mode)
		else:
			dec1 = F.max_unpool2d(
				dec2, indices_1, kernel_size=2, stride=2, output_size=dim_0
			)
		if self.skip_type is not None:
			if 'mul' in self.skip_type.lower():
				dec1*=enc1
			if 'add' in self.skip_type.lower():
				dec1+=enc1
		dec1 = self.dec11(dec1)
		dec1 = self.dec10(dec1)
		if self.BR_bool:
			dec1=self.BR(dec1)


		dec_softmax = F.softmax(dec1, dim=1)
		if self.latent:
			return enc3, dec1, dec_softmax
		else:
			return enc3, dec1, dec_softmax

class SegNet(nn.Module):
	def __init__(self, input_channels, output_channels):
		super(SegNet, self).__init__()

		self.input_channels = input_channels
		self.output_channels = output_channels

		self.num_channels = input_channels

		# self.vgg16 = models.vgg16(pretrained=True)


		# Encoder layers

		self.encoder_conv_00 = nn.Sequential(*[
												nn.Conv2d(in_channels=self.input_channels,
														  out_channels=64,
														  kernel_size=3,
														  padding=1),
												nn.BatchNorm2d(64)
												])
		self.encoder_conv_01 = nn.Sequential(*[
												nn.Conv2d(in_channels=64,
														  out_channels=64,
														  kernel_size=3,
														  padding=1),
												nn.BatchNorm2d(64)
												])
		self.encoder_conv_10 = nn.Sequential(*[
												nn.Conv2d(in_channels=64,
														  out_channels=128,
														  kernel_size=3,
														  padding=1),
												nn.BatchNorm2d(128)
												])
		self.encoder_conv_11 = nn.Sequential(*[
												nn.Conv2d(in_channels=128,
														  out_channels=128,
														  kernel_size=3,
														  padding=1),
												nn.BatchNorm2d(128)
												])
		self.encoder_conv_20 = nn.Sequential(*[
												nn.Conv2d(in_channels=128,
														  out_channels=256,
														  kernel_size=3,
														  padding=1),
												nn.BatchNorm2d(256)
												])
		self.encoder_conv_21 = nn.Sequential(*[
												nn.Conv2d(in_channels=256,
														  out_channels=256,
														  kernel_size=3,
														  padding=1),
												nn.BatchNorm2d(256)
												])
		self.encoder_conv_22 = nn.Sequential(*[
												nn.Conv2d(in_channels=256,
														  out_channels=256,
														  kernel_size=3,
														  padding=1),
												nn.BatchNorm2d(256)
												])
		self.encoder_conv_30 = nn.Sequential(*[
												nn.Conv2d(in_channels=256,
														  out_channels=512,
														  kernel_size=3,
														  padding=1),
												nn.BatchNorm2d(512)
												])
		self.encoder_conv_31 = nn.Sequential(*[
												nn.Conv2d(in_channels=512,
														  out_channels=512,
														  kernel_size=3,
														  padding=1),
												nn.BatchNorm2d(512)
												])
		self.encoder_conv_32 = nn.Sequential(*[
												nn.Conv2d(in_channels=512,
														  out_channels=512,
														  kernel_size=3,
														  padding=1),
												nn.BatchNorm2d(512)
												])
		self.encoder_conv_40 = nn.Sequential(*[
												nn.Conv2d(in_channels=512,
														  out_channels=512,
														  kernel_size=3,
														  padding=1),
												nn.BatchNorm2d(512)
												])
		self.encoder_conv_41 = nn.Sequential(*[
												nn.Conv2d(in_channels=512,
														  out_channels=512,
														  kernel_size=3,
														  padding=1),
												nn.BatchNorm2d(512)
												])
		self.encoder_conv_42 = nn.Sequential(*[
												nn.Conv2d(in_channels=512,
														  out_channels=512,
														  kernel_size=3,
														  padding=1),
												nn.BatchNorm2d(512)
												])

		# self.init_vgg_weigts()

		# Decoder layers

		self.decoder_convtr_42 = nn.Sequential(*[
												nn.ConvTranspose2d(in_channels=512,
																   out_channels=512,
																   kernel_size=3,
																   padding=1),
												nn.BatchNorm2d(512)
											   ])
		self.decoder_convtr_41 = nn.Sequential(*[
												nn.ConvTranspose2d(in_channels=512,
																   out_channels=512,
																   kernel_size=3,
																   padding=1),
												nn.BatchNorm2d(512)
											   ])
		self.decoder_convtr_40 = nn.Sequential(*[
												nn.ConvTranspose2d(in_channels=512,
																   out_channels=512,
																   kernel_size=3,
																   padding=1),
												nn.BatchNorm2d(512)
											   ])
		self.decoder_convtr_32 = nn.Sequential(*[
												nn.ConvTranspose2d(in_channels=512,
																   out_channels=512,
																   kernel_size=3,
																   padding=1),
												nn.BatchNorm2d(512)
											   ])
		self.decoder_convtr_31 = nn.Sequential(*[
												nn.ConvTranspose2d(in_channels=512,
																   out_channels=512,
																   kernel_size=3,
																   padding=1),
												nn.BatchNorm2d(512)
											   ])
		self.decoder_convtr_30 = nn.Sequential(*[
												nn.ConvTranspose2d(in_channels=512,
																   out_channels=256,
																   kernel_size=3,
																   padding=1),
												nn.BatchNorm2d(256)
											   ])
		self.decoder_convtr_22 = nn.Sequential(*[
												nn.ConvTranspose2d(in_channels=256,
																   out_channels=256,
																   kernel_size=3,
																   padding=1),
												nn.BatchNorm2d(256)
											   ])
		self.decoder_convtr_21 = nn.Sequential(*[
												nn.ConvTranspose2d(in_channels=256,
																   out_channels=256,
																   kernel_size=3,
																   padding=1),
												nn.BatchNorm2d(256)
											   ])
		self.decoder_convtr_20 = nn.Sequential(*[
												nn.ConvTranspose2d(in_channels=256,
																   out_channels=128,
																   kernel_size=3,
																   padding=1),
												nn.BatchNorm2d(128)
											   ])
		self.decoder_convtr_11 = nn.Sequential(*[
												nn.ConvTranspose2d(in_channels=128,
																   out_channels=128,
																   kernel_size=3,
																   padding=1),
												nn.BatchNorm2d(128)
											   ])
		self.decoder_convtr_10 = nn.Sequential(*[
												nn.ConvTranspose2d(in_channels=128,
																   out_channels=64,
																   kernel_size=3,
																   padding=1),
												nn.BatchNorm2d(64)
											   ])
		self.decoder_convtr_01 = nn.Sequential(*[
												nn.ConvTranspose2d(in_channels=64,
																   out_channels=64,
																   kernel_size=3,
																   padding=1),
												nn.BatchNorm2d(64)
											   ])
		self.decoder_convtr_00 = nn.Sequential(*[
												nn.ConvTranspose2d(in_channels=64,
																   out_channels=self.output_channels,
																   kernel_size=3,
																   padding=1)
											   ])

		self.latent = nn.Conv2d(in_channels = 512,
								out_channels = 256,
								kernel_size=4,
								stride=2,
								)
		self.tanh = nn.Tanh()

	def forward(self, input_img):
		"""
		Forward pass `input_img` through the network
		"""

		# Encoder

		# Encoder Stage - 1
		dim_0 = input_img.size()
		x_00 = F.relu(self.encoder_conv_00(input_img))
		x_01 = F.relu(self.encoder_conv_01(x_00))
		x_0, indices_0 = F.max_pool2d(x_01, kernel_size=2, stride=2, return_indices=True)

		# Encoder Stage - 2
		dim_1 = x_0.size()
		x_10 = F.relu(self.encoder_conv_10(x_0))
		x_11 = F.relu(self.encoder_conv_11(x_10))
		x_1, indices_1 = F.max_pool2d(x_11, kernel_size=2, stride=2, return_indices=True)

		# Encoder Stage - 3
		dim_2 = x_1.size()
		x_20 = F.relu(self.encoder_conv_20(x_1))
		x_21 = F.relu(self.encoder_conv_21(x_20))
		x_22 = F.relu(self.encoder_conv_22(x_21))
		x_2, indices_2 = F.max_pool2d(x_22, kernel_size=2, stride=2, return_indices=True)

		# Encoder Stage - 4
		dim_3 = x_2.size()
		x_30 = F.relu(self.encoder_conv_30(x_2))
		x_31 = F.relu(self.encoder_conv_31(x_30))
		x_32 = F.relu(self.encoder_conv_32(x_31))
		x_3, indices_3 = F.max_pool2d(x_32, kernel_size=2, stride=2, return_indices=True)

		# Encoder Stage - 5
		dim_4 = x_3.size()
		x_40 = F.relu(self.encoder_conv_40(x_3))
		x_41 = F.relu(self.encoder_conv_41(x_40))
		x_42 = F.relu(self.encoder_conv_42(x_41))
		x_4, indices_4 = F.max_pool2d(x_42, kernel_size=2, stride=2, return_indices=True)

		# Decoder

		dim_d = x_4.size()

		# Decoder Stage - 5
		x_4d = F.max_unpool2d(x_4, indices_4, kernel_size=2, stride=2, output_size=dim_4)
		x_42d = F.relu(self.decoder_convtr_42(x_4d))
		x_41d = F.relu(self.decoder_convtr_41(x_42d))
		x_40d = F.relu(self.decoder_convtr_40(x_41d))
		dim_4d = x_40d.size()

		# Decoder Stage - 4
		x_3d = F.max_unpool2d(x_40d, indices_3, kernel_size=2, stride=2, output_size=dim_3)
		x_32d = F.relu(self.decoder_convtr_32(x_3d))
		x_31d = F.relu(self.decoder_convtr_31(x_32d))
		x_30d = F.relu(self.decoder_convtr_30(x_31d))
		dim_3d = x_30d.size()

		# Decoder Stage - 3
		x_2d = F.max_unpool2d(x_30d, indices_2, kernel_size=2, stride=2, output_size=dim_2)
		x_22d = F.relu(self.decoder_convtr_22(x_2d))
		x_21d = F.relu(self.decoder_convtr_21(x_22d))
		x_20d = F.relu(self.decoder_convtr_20(x_21d))
		dim_2d = x_20d.size()

		# Decoder Stage - 2
		x_1d = F.max_unpool2d(x_20d, indices_1, kernel_size=2, stride=2, output_size=dim_1)
		x_11d = F.relu(self.decoder_convtr_11(x_1d))
		x_10d = F.relu(self.decoder_convtr_10(x_11d))
		dim_1d = x_10d.size()

		# Decoder Stage - 1
		x_0d = F.max_unpool2d(x_10d, indices_0, kernel_size=2, stride=2, output_size=dim_0)
		x_01d = F.relu(self.decoder_convtr_01(x_0d))
		x_00d = self.decoder_convtr_00(x_01d)
		dim_0d = x_00d.size()

		# x_softmax = F.softmax(x_00d, dim=1)

		#print("dim_0: {}".format(dim_0))
		#print("dim_1: {}".format(dim_1))
		#print("dim_2: {}".format(dim_2))
		#print("dim_3: {}".format(dim_3))
		#print("dim_4: {}".format(dim_4))
		#
		#print("dim_d: {}".format(dim_d))
		# print("dim_4d: {}".format(dim_4d))
		# print("dim_3d: {}".format(dim_3d))
		# print("dim_2d: {}".format(dim_2d))
		# print("dim_1d: {}".format(dim_1d))
		# print("dim_0d: {}".format(dim_0d))

		img_latent = self.latent(x_4)
		
		img_latent_softmax = F.softmax(img_latent, dim=1)
		x_recons = self.tanh(x_00d)

		return img_latent_softmax, x_recons,[x_0, x_1, x_2, x_3]

def parse_list(option, opt, value, parser):
	setattr(parser.values, option.dest, value.split(","))


if __name__ == "__main__":
	from graphviz import Digraph
	from torchviz import make_dot

	model = SegNet(input_channels=3, output_channels=3)#.to("cuda")
	#summary(model, input_size=(3, 128, 128))

	img = Variable(torch.randn(8, 3, 128, 128))#.to("cuda")
	latent, out, featlist = model(img)
	#graph = make_dot(out)
	# graph.view()
	print(latent.shape, out.shape, len(featlist))
	#cv2.imshow("color",out.detach().numpy())
