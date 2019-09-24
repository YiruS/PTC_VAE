'''
Generator(pointnet) for point cloud

author: Yefan
created: 8/8/19 11:21 PM
'''
import torch
import torch.nn as nn
import torch.nn.functional as Functional
import numpy as np
from encoders import Encoder
from dataset import PointCloudDataset
from pointnet import PointwiseMLP
import argparse
import sys

class GeneratorSingle(nn.Module):
	def __init__(self, dims):
		
		super(GeneratorSingle, self).__init__()
		self.mlp = PointwiseMLP(dims, doLastRelu=False)

	def forward(self, X):
		return self.mlp.forward(X)



class GeneratorVanilla(nn.Module):

	def __init__(self, Conv2d_dims, FC_dims, grid_dims, Generate1_dims,
		         Generate2_dims, MLP_doLastRelu=False):
	    
	    super(GeneratorVanilla,self).__init__()
	    N = grid_dims[0]*grid_dims[1]
	    u = (torch.arange(0., grid_dims[0]) / grid_dims[0]- 0.5).repeat(grid_dims[1])
	    v = (torch.arange(0., grid_dims[1]) / grid_dims[1] - 0.5).expand(grid_dims[0], -1).t().reshape(-1)
	    t = torch.empty(grid_dims[0]*grid_dims[1], dtype = torch.float)
	    t.fill_(0.)
	    self.encoder = Encoder(Conv2d_dims=Conv2d_dims,
                       FC_dims =FC_dims)
	    self.grid = torch.stack((u, v, t), 1)
	    self.N = grid_dims[0] * grid_dims[1]
	    self.G1 = GeneratorSingle(Generate1_dims)
	    self.G2 = GeneratorSingle(Generate2_dims)
	    
		  
	def forward(self, X):
		#self.vis()
		img_feat = self.encoder(X)                     # B* 512 
		img_feat = img_feat.unsqueeze(1)                  # B* 1 * 512
		codeword = img_feat.expand(-1, self.N, -1)      # B* self.N *512
		#print(codeword.shape)

		B = codeword.shape[0]                   # extract batch size
		tmpGrid = self.grid
		tmpGrid = tmpGrid.unsqueeze(0)
		tmpGrid = tmpGrid.expand(B, -1, -1).cuda()     # BxNx2
		
		#1st generating
		f = torch.cat((tmpGrid, codeword), 2)
		f = self.G1.forward(f)
		
		#2st generating
		f = torch.cat((f, codeword), 2)
		f = self.G2.forward(f)
		
		return f

def main(args):
	kwargs = {'num_workers':1, 'pin_memory':True} if args.cuda else {}
	generator = GeneratorVanilla(Conv2d_dims=(3, 96, 128, 192, 256, 512),
		FC_dims =(2048, 1024, 512),
        grid_dims=(64,32,1),
        Generate1_dims=(515,512,512,128),
        Generate2_dims=(640,512,512,3))

	dataloader = torch.utils.data.DataLoader(
                        PointCloudDataset(args.jsonfile_pkl, args.parent_dir, args.image_size),
                        batch_size=args.batch_size, shuffle= False, **kwargs)
	
	'''
	for x in dataloader:
		generator(x["image"])
		break 
	'''

if __name__ == '__main__':


	parser = argparse.ArgumentParser(sys.argv[0])
	parser.add_argument('-0','--jsonfile-pkl',type=str,
                        help='path of the jsonfile')
	parser.add_argument('-d','--parent-dir',type =str,
                        help ='path of data file')
	parser.add_argument('--image-size', type = int, default =128,
                        help ='image size ')
	parser.add_argument('--shuffle-point-order',type=str, default= 'no',
                         help ='whether/how to shuffle point order (no/offline/online)')
	parser.add_argument('--batch-size',type=int, default=16,
                        help='training batch size')
	parser.add_argument('--test-batch-size',type=int,default=32,
                        help='testing batch size')
	args = parser.parse_args(sys.argv[1:])
	args.cuda = torch.cuda.is_available()
	print(str(args))
	main(args)
	
    
	
	
  





