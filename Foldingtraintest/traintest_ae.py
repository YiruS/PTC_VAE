'''
Training stage for auto encoder 

author : Yefan
created: 8/10/19 12:00 AM
'''
import os
import sys
import argparse
import glog as logger
import torch
import torch.utils.data
import torch.autograd

from dataset import PointCloudDataset
from generator import GeneratorVanilla
from traintester import TrainTester, ChamfersDistance3
from utils import count_parameter_num




def main(args):


	kwargs = {'num_workers':1, 'pin_memory':True} if args.cuda else {}

        # load data
	train_loader = torch.utils.data.DataLoader(
                        PointCloudDataset(args.train_json, args.parent_dir),
                        batch_size=args.batch_size, shuffle= True, **kwargs)



	test_loader = torch.utils.data.DataLoader(
                        PointCloudDataset(args.test_json, args.parent_dir),

                        batch_size=args.test_batch_size, shuffle= True, **kwargs)


        # set model parameter num

	net = GeneratorVanilla(Conv2d_dims=(3, 96, 128, 192, 256, 512),
		FC_dims =(2048, 1024, 512),
        grid_dims=(64,32,1),
        Generate1_dims=(515,512,512,128),
        Generate2_dims=(640,512,512,3))


	logger.info('Number of parameters={}'.format(count_parameter_num(net.parameters())))
	solver = torch.optim.SGD(net.parameters(),
                   lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)



	loss_fn = ChamfersDistance3()


	# train and test

	runner = TrainTester(
        net=net, solver=solver, total_epochs=args.epoch,
        cuda=args.cuda, log_dir=args.log_dir, verbose_per_n_batch=args.verbose_per_n_batch
    )

	runner.run(train_loader=train_loader, test_loader= test_loader, loss_fn=loss_fn)
	

	logger.info('Done!')





if __name__ == "__main__":
	parser = argparse.ArgumentParser(sys.argv[0], description='Training Encoder_decoder')

	parser.add_argument('-0','--train-json',type=str,
                        help='path of the train json file')
	parser.add_argument('-1','--test-json',type=str,
                    help='path of the testing json file')
	parser.add_argument('-d','--parent-dir',type =str,
                        help ='path of data file')
	parser.add_argument('-e','--epoch',type=int,default=300,
                    help='training epochs')

	parser.add_argument('--batch-size',type=int,default=16,
                    help='training batch size')
	parser.add_argument('--test-batch-size',type=int,default=32,
                    help='testing batch size')
	parser.add_argument('--lr',type=float,default=1e-4,
                    help='learning rate')
	parser.add_argument('--momentum',type=float,default=0.9,
                    help='Solver momentum')
	parser.add_argument('--weight-decay',type=float,default=1e-6,
                    help='weight decay')
	parser.add_argument('--shuffle-point-order',type=str,default='no',
                    help='whether/how to shuffle point order (no/offline/online)')
	parser.add_argument('--log-dir',type=str,default='logs/tmp',
                    help='log folder to save training stats as numpy files')
	parser.add_argument('--verbose_per_n_batch',type=int,default=10,
                    help='log training stats to console every n batch (<=0 disables training log)')



	args = parser.parse_args(sys.argv[1:])
	args.cuda = torch.cuda.is_available()
	args.script_folder = os.path.dirname(os.path.abspath(__file__))


	print(str(args))
	sys.stdout.flush()
	main(args)













