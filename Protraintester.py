import os
import sys
sys.path.append("utils/chamfer/")
import glog as logger
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.utils import check_exist_or_mkdirs, make_D_label
from tensorboardX import SummaryWriter
from torch import autograd

from utils.visualize import pts2img
from utils.image_pool import ImagePool
from utils.loss import cal_gradient_penalty
import torchvision.utils as vutils
from sklearn.mixture import GaussianMixture
import tqdm

from chamfer_distance import ChamferDistance


'''
class ChamfersDistance(nn.Module):
	
	def __init__(self):
		super(ChamfersDistance, self).__init__()
		self.chamfer_dist = ChamferDistance()

	def forward(self,input1,input2):
		dist1, dist2 = self.chamfer_dist(input1,input2)
		loss = (torch.mean(dist1)) + (torch.mean(dist2))
		return loss
'''


def gradient_penalty(self, y, x):
		"""Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
		weight = torch.ones(y.size()).to(self.device)
		dydx = torch.autograd.grad(outputs=y,
								   inputs=x,
								   grad_outputs=weight,
								   retain_graph=True,
								   create_graph=True,
								   only_inputs=True)[0]

		dydx = dydx.view(dydx.size(0), -1)
		dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
		return torch.mean((dydx_l2norm-1)**2)

class ChamfersDistance(nn.Module):

	def __init__(self):
		super(ChamfersDistance, self).__init__()
		self.chamfer_dist = ChamferDistance()

	def forward(self, input1, input2):
		dist1, dist2 = self.chamfer_dist(input1, input2)
		loss = (torch.mean(dist1)) + (torch.mean(dist2))

		return loss





class Stats(object):
	def __init__(self):
		self.iter_loss = []

	def push_loss(self, iter, loss):
		self.iter_loss.append([iter, loss])

	def push(self, iter, loss):
		self.push_loss(iter, loss)

	def save(self, file):
		np.savez_compressed(
			file,
			iter_loss=np.asarray(self.iter_loss))



class TrainTester(object):

	def __init__(self, net, criterion_I, criterion_PTC, optimizer, lr_scheduler, logger, args):
		self.net = net
		self.criterion_I = criterion_I
		self.criterion_PTC = criterion_PTC
		self.optimizer = optimizer

		self.logger = logger

		self.lr_scheduler = lr_scheduler
		self.lr_schedulers = [self.lr_scheduler]

		self.total_epochs = args.total_epochs
		self.log_dir, self.verbose_per_n_batch = args.log_dir, args.verbose_per_n_batch
		self.save_model_per_n_batch = args.save_model_per_n_batch

		self.done = False
		self.train_iter = 0
		self.stats_train_batch = Stats()
		self.stats_train_running = Stats()
		self.stats_test = Stats()
		self.running_loss = None
		self.running_factor = 0.9
		self.epoch_callbacks = [self.save_stats]

		self.train_loss = float("inf")
		self.batch_size = args.batch_size
		self.npts = args.npts
		self.device = args.device

		self.grid = self.cube_generator(self.batch_size, self.npts, 3)

		self.lambda_image_loss = args.lambda_image_loss
		self.lambda_ptc_init = args.lambda_ptc_init
		self.lambda_ptc_prim = args.lambda_ptc_prim
		self.lambda_ptc_interm = args.lambda_ptc_interm
		self.lambda_ptc_fine = args.lambda_ptc_fine
		self.lambda_ptc_recons = args.lambda_ptc_recons
		self.lambda_kl = args.lambda_kl

		self.snapshot_dir = args.snapshot_dir
		self.log_dir = args.log_dir
		self.output_dir = args.output_dir
		check_exist_or_mkdirs(self.snapshot_dir)
		check_exist_or_mkdirs(self.log_dir)
		check_exist_or_mkdirs(self.output_dir)

		self.tensorboard = args.tensorboard
		if self.tensorboard:
			self.writer = SummaryWriter(self.log_dir)

		self.checkpoint = args.checkpoint
		self.testing = args.test
		if self.checkpoint:
			print("Loading checkpoint ...")
			self.net.load_state_dict(torch.load(self.checkpoint))

	def invoke_epoch_callback(self):
		if len(self.epoch_callbacks)>0:
			for ith, cb in enumerate(self.epoch_callbacks):
				try:
					cb()
				except:
					logger.warn('epoch_callback[{}] failed.'.format(ith))

	def adjust_lr_linear(self, step, total_step):
		base_lr = self.solver.defaults['lr']
		lr = base_lr * (total_step - step + 1.) / total_step
		for param_group in self.solver.param_groups:
			param_group['lr'] = lr

	def gaussian_generator(self, B, N, D):
		noise = torch.FloatTensor(B, N, D)
		for i in range(B):
			if D == 3:
			# set gaussian ceters and covariances in 3D
				means = np.array(
						[[0.0, 0.0, 0.0]]
						)
				covs = np.array([np.diag([0.01, 0.01, 0.03])
					 #np.diag([0.08, 0.01, 0.01]),
					 #np.diag([0.01, 0.05, 0.01]),
					 #np.diag([0.03, 0.07, 0.01])
					 ])
				n_gaussians = means.shape[0]
			points = []
			for i in range(len(means)):
				x = np.random.multivariate_normal(means[i], covs[i], N )
				points.append(x)
			points = np.concatenate(points)
			#fit the gaussian model
			gmm = GaussianMixture(n_components=n_gaussians, covariance_type='diag')
			gmm.fit(points)
			noise[i] = torch.tensor(points,dtype= torch.float)

		return noise

	def cube_generator(self, B, N, D):
		cube = torch.FloatTensor(B,N,D)
		x_count = 8
		y_count = 8
		z_count = 16
		count = x_count * y_count * z_count
		for i in range(B):
			one = np.zeros((count, 3))
			# setting vertices
			for t in range(count):
				x = float(t % x_count) / (x_count - 1)
				y = float((t / x_count) % y_count) / (y_count - 1)
				z = float(t / (x_count * y_count) % z_count) / (z_count - 1)
				one[t] = [x - 0.5, y - 0.5, z -0.5]
			one *= 0.5/one.max()
			cube[i]= torch.tensor(one,dtype= torch.float)
		return cube

	def train(self, epoch, loader):
		loss_sum, loss_sum_ptc, loss_sum_image, batch_loss = 0.0, 0.0, 0.0, 0.0

		loss_sum_kl = 0
		loss_sum_ptc_init, loss_sum_ptc_prim, loss_sum_ptc_interm, loss_sum_ptc_fine = 0, 0, 0, 0
		loss_sum_ptc_recons, loss_sum_image = 0, 0

		self.net.train()
		for batch_idx, batch in enumerate(loader):
			map(lambda scheduler: scheduler.step(), self.lr_schedulers)

			image, ptcloud = batch['image'], batch['ptcloud']
			proMatrix = batch['proMatrix']
			mask, category = batch['mask'], batch['category']
			image, ptcloud, mask = \
				Variable(image).to(self.device),\
				Variable(ptcloud).to(self.device), \
				Variable(mask, requires_grad=False).to(self.device)
			proMatrix = Variable(proMatrix,requires_grad=False).to(self.device)
			category = Variable(category, requires_grad=False).to(self.device)
			B = image.shape[0]

			if B != self.batch_size:
				grid = self.cube_generator(B,self.npts,3)
				gridv = Variable(grid).to(self.device)
			else:
				gridv = Variable(self.grid).to(self.device)

			self.optimizer.zero_grad()
			ptcloud_init, ptcloud_prim, ptcloud_interm, ptcloud_fine, img_recons, z_mu, z_sigma, _, _ = \
				self.net(
					x=image,
					grid=gridv,
					proMatrix=proMatrix,
					category=category,
				)

			## image reconstruction loss: mask L1 ##
			loss_image_recons = self.criterion_I(image, img_recons, mask)
			## ptc loss: chamfer distance ##
			loss_ptc_init = self.criterion_PTC(ptcloud_init, ptcloud)
			loss_ptc_interm = self.criterion_PTC(ptcloud_interm, ptcloud)
			loss_ptc_fine = self.criterion_PTC(ptcloud_fine, ptcloud)
			## VAE loss ##
			# https://arxiv.org/abs/1312.6114 (Appendix B)
			# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
			# KLD_element = z_mu.pow(2).add_(z_logvar.exp()).mul_(-1).add_(1).add_(z_logvar)
			# loss_KL = torch.sum(KLD_element).mul_(-0.5)
			KLD_element = torch.pow(z_mu, 2)+torch.pow(z_sigma, 2)-torch.log(1e-8 + torch.pow(z_sigma, 2))-1.0
			loss_KL = 0.5 * torch.sum(KLD_element)
			## reconstruction loss ##
			loss_ptc_recons = self.criterion_PTC(ptcloud_prim, ptcloud)

			loss_all = self.lambda_image_loss*loss_image_recons + \
					   self.lambda_ptc_init*loss_ptc_init + \
					   self.lambda_ptc_interm*loss_ptc_interm + \
					   self.lambda_ptc_fine*loss_ptc_fine + \
					   self.lambda_ptc_recons*loss_ptc_recons + \
					   self.lambda_kl*loss_KL

			loss_all.backward()
			self.optimizer.step()

			batch_loss_image = self.lambda_image_loss*loss_image_recons.item()
			batch_loss_ptc_init = self.lambda_ptc_init*loss_ptc_init.item()
			batch_loss_ptc_interm = self.lambda_ptc_interm*loss_ptc_interm.item()
			batch_loss_ptc_fine = self.lambda_ptc_fine*loss_ptc_fine.item()
			batch_loss_ptc_recons = self.lambda_ptc_recons*loss_ptc_recons.item()
			batch_loss_kl = self.lambda_kl*loss_KL
			batch_loss_ptc_all = batch_loss_ptc_init + \
								 batch_loss_ptc_interm + \
								 batch_loss_ptc_fine + \
								 batch_loss_ptc_recons
			loss_sum_kl += batch_loss_kl
			loss_sum_ptc_init += batch_loss_ptc_init
			loss_sum_ptc_interm += batch_loss_ptc_interm
			loss_sum_ptc_fine += batch_loss_ptc_fine
			loss_sum_ptc_recons += batch_loss_ptc_recons
			loss_sum_image += batch_loss_image
			loss_sum_ptc += batch_loss_ptc_all
			# if self.running_loss is None:
			# 	self.running_loss = batch_loss_ptc_all
			# else:
			# 	self.running_loss = self.running_factor*self.running_loss \
			# 						+ (1-self.running_factor)*batch_loss_ptc_all

			# collect stats
			self.train_iter += 1
			self.stats_train_batch.push(self.train_iter, loss=batch_loss)
			self.stats_train_running.push(self.train_iter, loss=self.running_loss)

			# logger
			if self.verbose_per_n_batch>0 and batch_idx % self.verbose_per_n_batch==0:
				self.logger.info((
					'Epoch={:<3d} [{:3.0f}%/{:<5d}] '
					'PTAll={:.3f} '
					'PTInit={:.3f} '
					'PTIntm={:.3f} '
					'PTFine={:.3f} '
					'PTRecons={:.3f} '
					'KL={:.3f} '
					'IMloss={:.3f}').format(
					epoch, 100.*batch_idx/loader.__len__(), len(loader.dataset),
					batch_loss_ptc_all,
					batch_loss_ptc_init,
					batch_loss_ptc_interm,
					batch_loss_ptc_fine,
					batch_loss_ptc_recons,
					batch_loss_kl,
					batch_loss_image,)
				)

			if self.tensorboard:
				scalar_info = {
					'PTAll': batch_loss_ptc_all,
					'PTInit': batch_loss_ptc_init,
					'PTIntm': batch_loss_ptc_interm,
					'PTFine': batch_loss_ptc_fine,
					'PTRecons': batch_loss_ptc_recons,
					'KL': batch_loss_kl,
					'IMloss': batch_loss_image,
				}

				for key, val in scalar_info.items():
					self.writer.add_scalar(key, val, self.train_iter)

		self.logger.info("======== Epoch {:<3d} ========".format(epoch))
		self.logger.info("Train: overall={:.3f}, genPT={:.3f}, reconsPT={:.3f}, image={:.3f}, KL={:.3f})".format(
			loss_sum_ptc / float(len(loader)),
			(loss_sum_ptc_init+loss_sum_ptc_interm+loss_sum_ptc_fine) / (3.0*float(len(loader))),
			loss_sum_ptc_recons / float(len(loader)),
			loss_sum_image / float(len(loader)),
			loss_sum_kl / float(len(loader)))
		)
		return (loss_sum_ptc_init+loss_sum_ptc_interm+loss_sum_ptc_fine) / (3.0*float(len(loader)))


	def test(self, epoch, loader):
		self.net.eval()
		test_loss = 0.
		counter = 0.
		batch_idx = 0

		for batch in tqdm.tqdm(loader, total=loader.__len__()):
			image, ptcloud = batch['image'], batch['ptcloud']
			proMatrix = batch['proMatrix']
			mask, category = batch['mask'], batch['category']
			image, ptcloud, mask = \
				Variable(image).to(self.device), \
				Variable(ptcloud).to(self.device), \
				Variable(mask, requires_grad=False).to(self.device)

			proMatrix = Variable(proMatrix, requires_grad=False).to(self.device)
			category = Variable(category, requires_grad=False).to(self.device)

			B = image.shape[0]
			if B != self.batch_size:
				grid = self.cube_generator(B, self.npts,3)
				gridv = Variable(grid).to(self.device)
			else:
				gridv = Variable(self.grid).to(self.device)

			with torch.set_grad_enabled(False):
				ptcloud_init, ptcloud_prim, ptcloud_interm, ptcloud_fine, img_recons, z_mu, z_logvar, h, w = \
					self.net(
						x=image,
						grid=gridv,
						proMatrix=proMatrix,
						category=category,
					)

			if batch_idx == 0:

				batch_ptcloud_gt, batch_ptcloud_init, batch_ptcloud_prim, batch_ptcloud_interm, batch_ptcloud_fine = \
					ptcloud.cpu().numpy(), \
					ptcloud_init.cpu().numpy(), \
					ptcloud_prim.cpu().numpy(), \
					ptcloud_interm.cpu().numpy(), \
					ptcloud_fine.cpu().numpy()
				im, h, w = image.cpu().numpy(), h.cpu().numpy(), w.cpu().numpy()

				np.save('%s/ptc_GT_%03d.npy' % (self.output_dir, epoch), batch_ptcloud_gt)
				np.save('%s/ptc_init_%03d.npy' % (self.output_dir, epoch), batch_ptcloud_init)
				np.save('%s/ptc_prim_%03d.npy' % (self.output_dir, epoch), batch_ptcloud_prim)
				np.save('%s/ptc_intm_%03d.npy' % (self.output_dir, epoch), batch_ptcloud_interm)
				np.save('%s/ptc_fine_%03d.npy' % (self.output_dir, epoch), batch_ptcloud_fine)
				np.save('%s/im_%03d.npy' % (self.output_dir, epoch), im)
				np.save('%s/h_%03d.npy' % (self.output_dir, epoch), h)
				np.save('%s/w_%03d.npy' % (self.output_dir, epoch), w)

				vutils.save_image(image.data,
								  '%s/real_images_epoch_%03d.jpeg' % (
									  self.output_dir,
									  epoch,
								  ), normalize=True)
				vutils.save_image(img_recons.data,
								  '%s/recons_images_epoch_%03d.jpeg' % (
									  self.output_dir,
									  epoch,
								  ), normalize=True)

				for ins in np.arange(1):
					pc_gt, pc_init, pc_prim, pc_intm, pc_fine = \
						ptcloud.cpu().numpy()[batch_idx,:,:], \
						ptcloud_init.cpu().numpy()[ins,:,:], \
						ptcloud_prim.cpu().numpy()[ins,:,:], \
						ptcloud_interm.cpu().numpy()[ins,:,:], \
						ptcloud_fine.cpu().numpy()[ins,:,:]

					fig = plt.figure(figsize=(10, 6))
					plt.clf
					ax = fig.add_subplot(231, projection='3d')
					ax.scatter(pc_gt[:, 0], pc_gt[:, 1], pc_gt[:, 2], marker='o')
					ax.set_xlim([-1, 1])
					ax.set_ylim([-1, 1])
					ax.set_zlim([-1, 1])
					ax.set_xlabel('X')
					ax.set_ylabel('Y')
					ax.set_zlabel('Z')
					ax.set_title("gt")
					ax = fig.add_subplot(232, projection='3d')
					ax.scatter(pc_init[:, 0], pc_init[:, 1], pc_init[:, 2], marker='o')
					ax.set_xlim([-1, 1])
					ax.set_ylim([-1, 1])
					ax.set_zlim([-1, 1])
					ax.set_xlabel('X')
					ax.set_ylabel('Y')
					ax.set_zlabel('Z')
					ax.set_title("Init")
					ax = fig.add_subplot(233, projection='3d')
					ax.scatter(pc_prim[:, 0], pc_prim[:, 1], pc_prim[:, 2], marker='o')
					ax.set_xlim([-1, 1])
					ax.set_ylim([-1, 1])
					ax.set_zlim([-1, 1])
					ax.set_xlabel('X')
					ax.set_ylabel('Y')
					ax.set_zlabel('Z')
					ax.set_title("Prim")
					ax = fig.add_subplot(235, projection='3d')
					ax.scatter(pc_intm[:, 0], pc_intm[:, 1], pc_intm[:, 2], marker='o')
					ax.set_xlim([-1, 1])
					ax.set_ylim([-1, 1])
					ax.set_zlim([-1, 1])
					ax.set_xlabel('X')
					ax.set_ylabel('Y')
					ax.set_zlabel('Z')
					ax.set_title("Intm")
					ax = fig.add_subplot(236, projection='3d')
					ax.scatter(pc_fine[:, 0], pc_fine[:, 1], pc_fine[:, 2], marker='o')
					ax.set_xlim([-1, 1])
					ax.set_ylim([-1, 1])
					ax.set_zlim([-1, 1])
					ax.set_xlabel('X')
					ax.set_ylabel('Y')
					ax.set_zlabel('Z')
					ax.set_title("Fine")
					if self.testing:
						title = os.path.join(self.output_dir, "Test_epoch_{}.png".format(epoch))
					else:
						title = os.path.join(self.output_dir, "Train_epoch_{}.png".format(epoch))
					fig.savefig(title)
					plt.close()

			#print(batch_idx)
			batch_idx += 1
			counter += 1
			loss_ptc_init = self.criterion_PTC(ptcloud_init, ptcloud)
			loss_ptc_interm = self.criterion_PTC(ptcloud_interm, ptcloud)
			loss_ptc_fine = self.criterion_PTC(ptcloud_fine, ptcloud)
			test_loss += self.lambda_ptc_init*loss_ptc_init.item() + \
						 self.lambda_ptc_interm*loss_ptc_interm.item() + \
						 self.lambda_ptc_fine*loss_ptc_fine.item()


		test_loss = test_loss / float(counter)
		self.stats_test.push(self.train_iter, loss=test_loss)
		self.logger.info('Test set (epoch={:<3d}): AverageLoss={:.4f}'.format(epoch, test_loss))


	def save_stats(self):
		self.stats_train_running.save(os.path.join(self.log_dir, 'stats_train_running.npz'))
		self.stats_train_batch.save(os.path.join(self.log_dir, 'stats_train_batch.npz'))
		self.stats_test.save(os.path.join(self.log_dir, 'stats_test.npz'))


	def run(self, train_loader, test_loader):
		self.net.to(self.device)

		self.logger.info('Network Architecture:')
		print(str(self.net))
		sys.stdout.flush()

		self.logger.info('Initial testing ...')
		self.test(epoch=0, loader=test_loader)
		for epoch in range(1, self.total_epochs+1):
			new_train_loss = self.train(
				epoch=epoch,
				loader=train_loader,
			)

			if new_train_loss < self.train_loss:
				self.logger.info('saving checkpoint ....')
				torch.save(self.net.state_dict(), os.path.join(self.snapshot_dir, "model_train_best.pth"))
				torch.save(self.optimizer.state_dict(), os.path.join(self.snapshot_dir,"solver_train_best.pth"))
				self.train_loss = new_train_loss
			if epoch % self.save_model_per_n_batch == 0:
				self.test(
					epoch=epoch,
					loader=test_loader
				)
				self.invoke_epoch_callback()
		self.save_stats()

		self.done=True

