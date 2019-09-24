import numpy as np
import torch
import torch.nn as nn

# # Define Chamfer Loss
# import sys
# sys.path.append("utils/chamfer/")
# import dist_chamfer as ext
# distChamfer = ext.chamferDist()


class MaskedL1(nn.Module):
	def __init__(self):
		super(MaskedL1, self).__init__()
		self.criterion = nn.L1Loss(reduction="sum")

	def forward(self, gt, pred, mask):
		loss = self.criterion(gt*mask, pred*mask)
		loss /= (mask==1.0).sum()
		return loss

def make_D_label(input, value, random=False):
	if random:
		if value == 0:
			lower, upper = 0, 0.205
		elif value ==1:
			lower, upper = 0.8, 1.05
		D_label = torch.FloatTensor(input.data.size()).uniform_(lower, upper)
	else:
		D_label = torch.FloatTensor(input.data.size()).fill_(value)

	return D_label

class GANLoss(nn.Module):
	"""Define different GAN objectives.
	The GANLoss class abstracts away the need to create the target label tensor
	that has the same size as the input.
	"""

	def __init__(self, gan_mode, device, target_real_label=1.0, target_fake_label=0.0):
		""" Initialize the GANLoss class.
		Parameters:
			gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
			target_real_label (bool) - - label for a real image
			target_fake_label (bool) - - label of a fake image
		Note: Do not use sigmoid as the last layer of Discriminator.
		LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
		"""
		super(GANLoss, self).__init__()
		self.register_buffer('real_label', torch.tensor(target_real_label))
		self.register_buffer('fake_label', torch.tensor(target_fake_label))
		self.gan_mode, self.device = gan_mode, device
		if gan_mode == 'lsgan':
			self.loss = nn.MSELoss().to(device)
		elif gan_mode == 'vanilla':
			self.loss = nn.BCEWithLogitsLoss().to(device)
		elif gan_mode == 'wgan':
			self.loss = None
		else:
			raise NotImplementedError('gan mode %s not implemented' % gan_mode)

	def get_target_tensor(self, prediction, target_is_real, random=False):
		"""Create label tensors with the same size as the input.
		Parameters:
			prediction (tensor) - - tpyically the prediction from a discriminator
			target_is_real (bool) - - if the ground truth label is for real images or fake images
		Returns:
			A label tensor filled with ground truth label, and with the size of the input
		"""

		if target_is_real:
			target_tensor = make_D_label(
				input=prediction,
				value=self.real_label,
				random=random,
			)
		else:
			target_tensor = make_D_label(
				input=prediction,
				value=self.fake_label,
				random=random,
			)
		return target_tensor

	def __call__(self, prediction, target_is_real, random=False):
		"""Calculate loss given Discriminator's output and grount truth labels.
		Parameters:
			prediction (tensor) - - tpyically the prediction output from a discriminator
			target_is_real (bool) - - if the ground truth label is for real images or fake images
		Returns:
			the calculated loss.
		"""
		if self.gan_mode == 'lsgan':
			target_tensor = self.get_target_tensor(
				prediction=prediction,
				target_is_real=target_is_real,
				random=False,
			).to(self.device)
			loss = self.loss(prediction, target_tensor)
		elif self.gan_mode == 'vanilla':
			target_tensor = self.get_target_tensor(
				prediction=prediction,
				target_is_real=target_is_real,
				random=random,
			).to(self.device)
			loss = self.loss(prediction, target_tensor)
		elif self.gan_mode == 'wgan':
			if target_is_real:
				loss = -prediction.mean()
			else:
				loss = prediction.mean()
		return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
	"""Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
	Arguments:
		netD (network)              -- discriminator network
		real_data (tensor array)    -- real images
		fake_data (tensor array)    -- generated images from the generator
		device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
		type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
		constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
		lambda_gp (float)           -- weight for this loss
	Returns the gradient penalty loss
	"""
	if lambda_gp > 0.0:
		if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
			interpolatesv = real_data
		elif type == 'fake':
			interpolatesv = fake_data
		elif type == 'mixed':
			alpha = torch.rand(real_data.shape[0], 1, device=device)
			alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
			interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
		else:
			raise NotImplementedError('{} not implemented'.format(type))
		interpolatesv.requires_grad_(True)
		disc_interpolates, _ = netD(interpolatesv)
		gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
										grad_outputs=torch.ones(disc_interpolates.size()).to(device),
										create_graph=True, retain_graph=True, only_inputs=True)
		gradients = gradients[0].contiguous().view(real_data.size(0), -1)  # flat the data
		gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
		return gradient_penalty, gradients
	else:
		return 0.0, None

# def laplace_coord(input, lap_idx, block_id, use_cuda = True):
#
#     # Inputs :
#     # input : nodes Tensor, size (n_pts, n_features)
#     # lap_idx : laplace index matrix Tensor, size (n_pts, 10)
#     #
#     # Returns :
#     # The laplacian coordinates of input with respect to edges as in lap_idx
#
#
#     vertex = torch.cat((input, torch.zeros(1, 3).cuda()), 0) if use_cuda else torch.cat((input, torch.zeros(1, 3)), 0)
#
#     indices = torch.tensor(lap_idx[block_id][:, :8])
#     weights = torch.tensor(lap_idx[block_id][:,-1], dtype = torch.float32)
#
#     if use_cuda:
#         indices = indices.cuda()
#         weights = weights.cuda()
#
#     weights = torch.reciprocal(weights).reshape((-1, 1)).repeat((1, 3))
#
#     num_pts, num_indices = indices.shape[0], indices.shape[1]
#     indices = indices.reshape((-1,))
#     vertices = torch.index_select(vertex, 0, indices)
#     vertices = vertices.reshape((num_pts, num_indices, 3))
#
#     laplace = torch.sum(vertices, 1)
#     laplace = input - torch.mul(laplace, weights)
#
#     return laplace
#
# def laplace_loss(input1, input2, lap_idx, block_id, use_cuda = True):
#
#     # Inputs :
#     # input1, input2 : nodes Tensor before and after the deformation
#     # lap_idx : laplace index matrix Tensor, size (n_pts, 10)
#     # block_id : id of the deformation block (if different than 1 then adds
#     # a move loss as in the original TF code)
#     #
#     # Returns :
#     # The Laplacian loss, with the move loss as an additional term when relevant
#
#     lap1 = laplace_coord(input1, lap_idx, block_id, use_cuda)
#     lap2 = laplace_coord(input2, lap_idx, block_id, use_cuda)
#     laplace_loss = torch.mean(torch.sum(torch.pow(lap1 - lap2, 2), 1)) * 1500
#     move_loss = torch.mean(torch.sum(torch.pow(input1 - input2, 2), 1)) * 100
#
#     if block_id == 0:
#         return laplace_loss
#     else:
#         return laplace_loss + move_loss
#
#
#
# def edge_loss(pred, gt_pts, edges, block_id, use_cuda = True):
#
# 	# edge in graph
#     #nod1 = pred[edges[block_id][:, 0]]
#     #nod2 = pred[edges[block_id][:, 1]]
#     idx1 = torch.tensor(edges[block_id][:, 0]).long()
#     idx2 = torch.tensor(edges[block_id][:, 1]).long()
#
#     if use_cuda:
#         idx1 = idx1.cuda()
#         idx2 = idx2.cuda()
#
#     nod1 = torch.index_select(pred, 0, idx1)
#     nod2 = torch.index_select(pred, 0, idx2)
#     edge = nod1 - nod2
#
# 	# edge length loss
#     edge_length = torch.sum(torch.pow(edge, 2), 1)
#     edge_loss = torch.mean(edge_length) * 300
#
#     return edge_loss
#
#
# def L1Tensor(img1, img2) :
# 	""" input shoudl be tensor and between 0 and 1"""
# 	mae = torch.mean(torch.abs(img2 - img1))
# 	return mae
#
#
# def L2Tensor(img1, img2) :
# 	""" input shoudl be tensor and between 0 and 1"""
# 	mse = torch.mean((img2 - img1) ** 2)
# 	return mse
#
# def chamfer_distance(gt, pred):
#
# 	dist1, dist2 = distChamfer(gt, pred) # BxN
# 	my_chamfer_loss = torch.mean(dist1, 1) + torch.mean(dist2, 1) # B
# 	loss_cd = torch.mean(my_chamfer_loss) # 1
# 	return loss_cd
#
#
# def total_pts_loss(pred_pts_list, pred_feats_list, gt_pts, ellipsoid, use_cuda = True):
#     """
#     pred_pts_list: [x1, x1_2, x2, x2_2, x3]
#     """
#
#     my_chamfer_loss, my_edge_loss, my_lap_loss = 0., 0., 0.
#     lap_const = [0.2, 1., 1.]
#
#     for i in range(3):
#         dist1, dist2 = distChamfer(gt_pts, pred_pts_list[i].unsqueeze(0))
#         my_chamfer_loss += torch.mean(dist1) + torch.mean(dist2)
#         my_edge_loss += edge_loss(pred_pts_list[i], gt_pts, ellipsoid["edges"], i, use_cuda)
#         my_lap_loss += lap_const[i] * laplace_loss(pred_feats_list[i], pred_pts_list[i], ellipsoid["lap_idx"], i, use_cuda)
#
#     my_pts_loss = 100 * my_chamfer_loss + 0.1 * my_edge_loss + 0.3 * my_lap_loss
#
#     return my_pts_loss
#
#
#
# def total_img_loss(pred_img, gt_img):
#
#     my_rect_loss = torch.nn.functional.binary_cross_entropy(pred_img, gt_img, size_average = False)
#     my_l1_loss = L1Tensor(pred_img, gt_img)
#
#     img_loss = my_rect_loss + my_l1_loss
#
#     return img_loss



