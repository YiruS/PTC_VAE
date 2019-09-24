'''
Training stage for auto encoder 

author : Yefan
created: 8/10/19 12:00 AM
'''
import os
import sys
import argparse
import glog as logger
import logging
import itertools

import torch
import torch.utils.data
import torch.autograd
import optparse
import time
from dataset.dataset import PointCloudDataset_Cached, PointCloudDataset
from model.ProGenerator import GeneratorVAE
from utils.utils import init_weights
from utils.loss import MaskedL1
from Protraintester import ChamfersDistance, TrainTester


def main(args):
		# load data
	starter_time = time.time()
	kwargs = {'num_workers':4, 'pin_memory':True}

	print("loading train data ...")
	trainset = PointCloudDataset_Cached(args.train_json)
	train_loader = torch.utils.data.DataLoader(
		trainset,
		batch_size=args.batch_size,
		shuffle= True,**kwargs,
	)
	print("loading test data ...")
	testset = PointCloudDataset_Cached(args.test_json)
	test_loader = torch.utils.data.DataLoader(
		testset,
		batch_size=args.test_batch_size,
		shuffle= True,**kwargs
	)
	print("Initialize cache={}".format(time.time()-starter_time))
	net = GeneratorVAE(
		encoder_dim=(3, 3),
		grid_dims=(32, 32, 1),
		Generate1_dims=259,
		Generate2_dims=1091,
		Generate3_dims=1219,
		args=args,
	)
	init_weights(net, init_type="xavier")

	logger = logging.getLogger()
	file_log_handler = logging.FileHandler(args.log_dir + args.log_filename)
	logger.addHandler(file_log_handler)

	stderr_log_handler = logging.StreamHandler(sys.stdout)
	logger.addHandler(stderr_log_handler)

	logger.setLevel('INFO')
	formatter = logging.Formatter()
	file_log_handler.setFormatter(formatter)
	stderr_log_handler.setFormatter(formatter)
	logger.info(args)

	criterion_I = MaskedL1().to(args.device)
	criterion_PTC = ChamfersDistance().to(args.device)
	optimizer = torch.optim.Adam(
		net.parameters(),
		lr=args.lr,
		betas=(args.adam_beta1, 0.999),
		weight_decay=args.weight_decay,
	)

	lr_scheduler = torch.optim.lr_scheduler.StepLR(
		optimizer,
		step_size=args.lr_decay_step,
		gamma=args.lr_decay_rate,
	)


	# train and test

	runner = TrainTester(
		net=net,
		criterion_I=criterion_I,
		criterion_PTC=criterion_PTC,
		optimizer=optimizer,
		lr_scheduler=lr_scheduler,
		logger=logger,
		args=args,
	)

	if args.train:
		runner.run(
			train_loader=train_loader,
			test_loader=test_loader,
		)
		logger.info('Training Done!')

	if args.test:
		runner.test(
			epoch=args.total_epochs + 1,
			loader=test_loader,
		)
		logger.info('Testing Done!')



if __name__ == "__main__":
	parser = optparse.OptionParser(sys.argv[0], description="Training Encoder_decoder")

	parser.add_option("--train-json",
					  dest="train_json", type=str,
					  default="data/pix3d/train_pix3d.json",
					  help='path of the training json file')
	parser.add_option("--test-json",
					  dest="test_json", type=str,
					  default="data/pix3d/test_pix3d.json",
					  help='path of the testing json file')
	parser.add_option("--checkpoint", type="str",
					  dest="checkpoint",
					  default=None,
					  help="Path to pretrained model")

	parser.add_option("--log-dir",
					  dest="log_dir", type=str,
					  default="logs/",
					  help="log folder to save training stats as numpy files")
	parser.add_option("--log-filename", type=str,
					  dest="log_filename",
					  default="Train.log",
					  help="Name of log file.")
	parser.add_option("--output-dir",
					  dest="output_dir", type=str,
					  default='results/',
					  help='result folder to save generated ptc during training')

	parser.add_option("--snapshot-dir",
					  dest="snapshot_dir", type=str,
					  default='snapshots/',
					  help='snapshot folder to save training checkpoint')
	parser.add_option('--verbose_per_n_batch',
					 dest="verbose_per_n_batch", type=int,
					 default=1,
					 help='log training stats to console every n batch (<=0 disables training log)')
	parser.add_option('--save_model_per_n_batch',
					  dest="save_model_per_n_batch", type=int,
					  default=5,
					  help='log training stats to console every n batch (<=0 disables training log)')
	parser.add_option("--shuffle-point-order", action="store_true",
					  dest="shuffle_point_order",
					  default=False,
					  help="whether/how to shuffle point order (no/offline/online)")
	parser.add_option('--parent-dir',
					 dest = 'data dir',type =str,
					 default= 'data/pix3d',
					 help ='path of data file')

	## training parameter
	parser.add_option("--total-epochs",
					  dest="total_epochs", type=int,
					default=500,
					help='training epochs')
	parser.add_option("--batch-size",
					  dest="batch_size", type=int,
					  default=4,
					help='training batch size')
	parser.add_option("--test-batch-size",
					  dest="test_batch_size", type=int,
					  default=4,
					help='testing batch size')
	parser.add_option("--lr",
					  dest="lr", type=float,
					  default=1e-4,
					help='learning rate')
	parser.add_option("--input-dim",
					  dest="input_dim", type=int,
					  default=3,
					  help='pts dim')
	parser.add_option("--zdim",
					  dest="zdim", type=int,
					  default=128,
					  help='dim for VAE latent')
	parser.add_option("--adam-beta1",
					  dest="adam_beta1", type=float,
					  default=0.5,
					  help="beta1 for Adam optimizer")
	parser.add_option("--npts",
					  dest="npts", type=int,
					  default=1024,
					  help="#points in grid")

	## hyper params ##
	parser.add_option("--lambda-image-loss",
					  dest="lambda_image_loss", type=float,
					  default=1.0,
					  help="lambda for image reconstruction loss")
	parser.add_option("--lambda-ptc-init",
					  dest="lambda_ptc_init", type=float,
					  default=1.0,
					  help="lambda for point cloud")
	parser.add_option("--lambda-ptc-prim",
					  dest="lambda_ptc_prim", type=float,
					  default=1.0,
					  help="lambda for point cloud")
	parser.add_option("--lambda-ptc-interm",
					  dest="lambda_ptc_interm", type=float,
					  default=1.0,
					  help="lambda for point cloud")
	parser.add_option("--lambda-ptc-fine",
					  dest="lambda_ptc_fine", type=float,
					  default=1.0,
					  help="lambda for point cloud")
	parser.add_option("--lambda-ptc-recons",
					  dest="lambda_ptc_recons", type=float,
					  default=1.0,
					  help="lambda for point cloud")
	parser.add_option("--lambda-kl",
					  dest="lambda_kl", type=float,
					  default=1.0,
					  help="lambda for VAE")

	## training settings ##
	parser.add_option('--momentum',type=float,
					  default=0.9,
					 help='Solver momentum')
	parser.add_option("--weight-decay",
					  dest="weight_decay", type=float,
					  default=1e-6,
					help='weight decay')
	parser.add_option('--lr_decay_step',
					  type=int, default=1762 * 4 * 20,
					  help='learning rate decay step'
					  )
	parser.add_option('--lr_decay_rate',
					  type=float, default=0.95,
					  help='learning rate decay rate'
					  )

	parser.add_option("--tensorboard", action="store_true",
					  dest="tensorboard",
					  default=False)
	parser.add_option("--train", action="store_true",
					  dest="train",
					  default=False,
					  help="run training", )
	parser.add_option("--test", action="store_true",
					  dest="test",
					  default=False,
					  help="run testing (generate ptcloud)", )

	(args, opts) = parser.parse_args()
   
	args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	args.script_folder = os.path.dirname(os.path.abspath(__file__))
	print(str(args))
	sys.stdout.flush()
	main(args)















