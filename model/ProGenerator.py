'''
Generator(pointnet) for point cloud

author: Yefan
created: 8/8/19 11:21 PM
'''
import os
import sys
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable

from graphviz import Digraph
from torchviz import make_dot

from model.SegNet import SegNet
from model.pointnet import PointwiseMLP
from model.Point_VAE import PointVAE

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/.." % file_path)
from utils.torchsummary import summary
from utils.utils import init_weights
import numpy as np
from dataset.dataset import PointCloudDataset
import torchvision.utils as vutils

sensor_width = 32

class Generator_Shared(nn.Module):
    def __init__(self, input_dim):
        super(Generator_Shared, self).__init__()

        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=512, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3)


class GeneratorSingle(nn.Module):
    def __init__(self, dims):
        
        super(GeneratorSingle, self).__init__()
        self.mlp = PointwiseMLP(dims, doLastRelu=False)

    def forward(self, X):
        return self.mlp.forward(X)


class PointGeneration(nn.Module):
    def __init__(self, dims):
        super(PointGeneration, self).__init__()
        layers = []
        for layer in range(len(dims)-1):
            layers.append(nn.Linear(dims[layer], dims[layer+1]))
            if layer != len(dims)-2:
                layers.append(nn.ReLU(True))
        self.pcg = nn.Sequential(*layers)

    def forward(self, x):
        return self.pcg(x)

class FoldingNet(nn.Module):
	def __init__(self, indim):
		super(FoldingNet, self).__init__()
		self.indim = indim
		self.fc1 = torch.nn.Linear(indim, 128)
		self.fc2 = torch.nn.Linear(128, 128)
		self.fc3 = torch.nn.Linear(128, 3)

		self.relu = torch.nn.ReLU()

	def forward(self, x):
		x = self.relu(self.fc1(x))
		x_feat = self.relu(self.fc2(x))
		x = self.fc3(x_feat)
		return x, x_feat

class PointProjection(nn.Module):
    """Graph Projection layer, which pool 2D features to mesh
    The layer projects a vertex of the mesh to the 2D image and use 
    bilinear interpolation to get the corresponding feature.
    """

    def __init__(self):
        super(PointProjection, self).__init__()


    def forward(self, img_features, input, proMatrix):
        
        B = input.shape[0]
        self.featlist = img_features
        ## transform: rot and translate
        ## cam intrinc            
        infill = torch.tensor(np.ones((B, input.shape[1], 1)),dtype =torch.float)          #BxNx1
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        infill = infill.to(device)
        input_4by1 = torch.transpose(torch.cat((input,infill),2),1,2)                     #BxNx4 -> Bx4xN                                           
        img_annotation = torch.bmm(proMatrix, input_4by1)                                 #Bx4xN 
        ##  x/z  y/z  
        img_annotation[:,0,:] = torch.div(img_annotation[:,0,:],img_annotation[:,2,:])    
        img_annotation[:,1,:] = torch.div(img_annotation[:,1,:],img_annotation[:,2,:])
        
                           
        w = torch.unsqueeze(img_annotation[:,0,:],2)
        h = torch.unsqueeze(img_annotation[:,1,:],2)
        
              
        h = torch.clamp(h,min = 0, max = 127)
        w = torch.clamp(w,min = 0, max = 127)
        
       
      
        feats = []
        img_sizes = [64, 32, 16, 8]
        out_dims = [64, 128, 256, 512]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
       
        for i in range(len(img_sizes)):
            #start.record()
            out = self.project(i, h, w, img_sizes[i], out_dims[i])
            #end.record()
            #torch.cuda.synchronize()
            
            #print('one projection',start.elapsed_time(end))  # milliseconds
            feats.append(out)
        #end.record()
        #torch.cuda.synchronize()

        #print('one projection',start.elapsed_time(end))  # milliseconds
        pixel_feature = torch.cat((feats[0],feats[1],feats[2],feats[3]),2)
        
        return pixel_feature, h, w

        


        

    def project(self, index, h, w, img_size, out_dim):
        #start = torch.cuda.Event(enable_timing=True)
        #end = torch.cuda.Event(enable_timing=True)
       
        img_feat = self.featlist[index]

        B = h.shape[0]
        num = h.shape[1]

        
        x = w / (128. / img_size)                                             # CHW ->   CYX
        y = h / (128. / img_size)
        

        x1, x2 = torch.floor(x).long(), torch.ceil(x).long()
        y1, y2 = torch.floor(y).long(), torch.ceil(y).long()
        
        x1 = torch.clamp(x1, max = img_size - 1)
        y1 = torch.clamp(y1, max = img_size - 1)
        x2 = torch.clamp(x2, max = img_size - 1)
        y2 = torch.clamp(y2, max = img_size - 1)

        #Q11 = torch.index_select(torch.index_select(img_feat, 1, x1), 1, y1)
        #Q12 = torch.index_select(torch.index_select(img_feat, 1, x1), 1, y2)
        #Q21 = torch.index_select(torch.index_select(img_feat, 1, x2), 1, y1)
        #Q22 = torch.index_select(torch.index_select(img_feat, 1, x2), 1, y2)
        output = torch.zeros(B, num, out_dim)
        
        #start.record()
        for i in range(B):
              
            Q11 = img_feat[i, :, y1[i], x1[i]].clone()
            Q12 = img_feat[i, :, y2[i], x1[i]].clone()
            Q21 = img_feat[i, :, y1[i], x2[i]].clone()
            Q22 = img_feat[i, :, y2[i], x2[i]].clone()
           
            Q11 = torch.squeeze(Q11)
            Q12 = torch.squeeze(Q12)
            Q21 = torch.squeeze(Q21)
            Q22 = torch.squeeze(Q22)
            
            #start.record()
            weights = torch.mul(x2[i] - x[i].long(), y2[i] - y[i].long())
         
            Q11 = torch.mul(weights.float(), torch.transpose(Q11, 0, 1))
            
            weights = torch.mul(x2[i] - x[i].long(), y[i].long() - y1[i])
            Q12 = torch.mul(weights.float(), torch.transpose(Q12, 0 ,1))

            weights = torch.mul(x[i].long() - x1[i], y2[i] - y[i].long())
            Q21 = torch.mul(weights.float(), torch.transpose(Q21, 0, 1))

            weights = torch.mul(x[i].long() - x1[i], y[i].long() - y1[i])
            Q22 = torch.mul(weights.float(), torch.transpose(Q22, 0, 1))
            #end.record()
            #torch.cuda.synchronize()

            #print('clone',start.elapsed_time(end))  # milliseconds
            output[i,:,:] = Q11 + Q21 + Q12 + Q22
        #end.record()
        #torch.cuda.synchronize()

        #print('each feature map',start.elapsed_time(end))  # milliseconds
        return output


class GeneratorVAE(nn.Module):

	def __init__(self,
				 pointVAE,
				 im_encoder,
				 encoder_dim,
				 grid_dims,
				 Generate1_dims,
				 Generate2_dims,
				 Generate3_dims,
				 args,
		):
		super(GeneratorVAE, self).__init__()
		self.args = args

		#self.im_encoder = SegNet(input_channels=encoder_dim[0], output_channels=encoder_dim[1])
		self.im_encoder = im_encoder
		init_weights(self.im_encoder, init_type="kaiming")
		self.N = grid_dims[0] * grid_dims[1]
		# self.G1 = PointGeneration(Generate1_dims)
		self.G1 = FoldingNet(indim=Generate1_dims)
		init_weights(self.G1, init_type="xavier")
		self.G2 = FoldingNet(indim=Generate2_dims)
		init_weights(self.G2, init_type="xavier")
		self.G3 = FoldingNet(indim=Generate3_dims)
		init_weights(self.G3, init_type="xavier")
		#self.pointVAE = PointVAE(args=args)
		self.pointVAE = pointVAE
		init_weights(self.pointVAE, init_type="xavier")

		self.P1 = PointProjection()
		self.P2 = PointProjection()

	def forward(self, x, grid, proMatrix, category):
		img_latent, img_recons, featlist = self.im_encoder(x)  # Bx256x1x1
		img_latent = ((img_latent.squeeze(2)).squeeze(2)).unsqueeze(1)  # Bx1x256
		codeword = img_latent.expand(-1, self.N, -1)  # BxNx256

		## 1st folding ##
		x = torch.cat((grid, codeword), dim=2) #BxNx(3+256)
		x_init, x_init_feat = self.G1(x)

		## VAE ##
		#x_primitive, z_mu, z_logvar = self.pointVAE(x_init)
		x_primitive, z_mu_x, z_sigma_x, z_mu_y, z_sigma_y, z_mu_z, z_sigma_z = self.pointVAE(x_init)

		## 1st projection ##
		pixel_feature_p1, h_p1, w_p1 = self.P1(featlist, x_primitive, proMatrix)
		pixel_feature_p1 = pixel_feature_p1.to(self.args.device)

		## 2nd folding ##
		x = torch.cat((x_primitive, pixel_feature_p1, x_init_feat), 2)  # BxNx(3+960+128)
		x_intermediate, x_interm_feat = self.G2.forward(x)

		## 2st projection ##
		pixel_feature_p2, h_p2, w_p2 = self.P1(featlist, x_intermediate, proMatrix)
		pixel_feature_p2 = pixel_feature_p2.to(self.args.device)

		# 2st generating
		x = torch.cat((x_intermediate, pixel_feature_p2, x_interm_feat, x_init_feat), 2)  # BxNx(3+960+128+128)
		x_fine, _ = self.G3.forward(x)

		#return x_init, x_primitive, x_intermediate, x_fine, img_recons, z_mu, z_logvar, h_p2, w_p2
		return x_init, x_primitive, x_intermediate, x_fine, img_recons, \
			   z_mu_x, z_sigma_x, z_mu_y, z_sigma_y, z_mu_z, z_sigma_z,\
			   h_p2, w_p2




class GeneratorVanilla(nn.Module):

    def __init__(self,
                 encoder_dim,
                 grid_dims,
                 Generate1_dims,
                 Generate2_dims
                    ):
        super(GeneratorVanilla, self).__init__()

        
        self.encoder = SegNet(input_channels=encoder_dim[0], output_channels=encoder_dim[1])
        init_weights(self.encoder, init_type="kaiming")
        self.N = grid_dims[0] * grid_dims[1]
        self.G1 = PointGeneration(Generate1_dims)

        init_weights(self.G1, init_type="xavier")
        self.G2 = PointGeneration(Generate2_dims)
        init_weights(self.G2, init_type="xavier")
        # self.reconstruct = nn.Tanh()

        self.P0 = PointProjection()
        self.P1 = PointProjection()

    def forward(self, x, noise, proMatrix):
        img_latent, img_recons, featlist = self.encoder(x)   # Bx256x1x1
        
        #img_reconst = self.reconstruct(img_feature)
        
        img_latent = ((img_latent.squeeze(2)).squeeze(2)).unsqueeze(1)     # Bx1x256
        codeword = img_latent.expand(-1, self.N, -1)                       # B x self.N x 256
        #1st projection
        
        pixel_feature_n, h_1, w_1 = self.P0(featlist, noise, proMatrix)
        
        pixel_feature_n = pixel_feature_n.to("cuda")
        #1st generating
        
        x = torch.cat((noise, codeword, pixel_feature_n), 2)                 # BxNx(256+3+ 960)
        x_primitive = self.G1.forward(x)                                   # BxNx3

        #2st   projection
        pixel_feature, h_2, w_2 = self.P1(featlist, x_primitive, proMatrix)
        pixel_feature = pixel_feature.to("cuda")
        
        #2st generating
        x = torch.cat((x_primitive, codeword, pixel_feature), 2)             # BxNx(3+ 256 + 960)
        x = self.G2.forward(x)
        
        return h_2, w_2, img_recons, x_primitive, x       


def main(args):
    kwargs = {'num_workers':1, 'pin_memory':True} if args.cuda else {}
    generator = GeneratorVanilla(
        encoder_dim = (3,3),
        grid_dims=(32,32,1),
        Generate1_dims=(1219,128,128,3),
        Generate2_dims=(1219,256,256,3))

    generator.to("cuda")
    dataloader = torch.utils.data.DataLoader(
                       PointCloudDataset(args.jsonfile_pkl, args.parent_dir, args.image_size),
                       batch_size=args.batch_size, shuffle= False, **kwargs)

    
    
    for x in dataloader:
        noise = x['ptcloud'].to("cuda")
        h, w, img_latent, pc1, pc2 = generator(x['image'].to("cuda"),noise,x['proMatrix'].to("cuda"))
        graph = make_dot(pc2)
        graph.engine = 'dot'
        graph.format = 'pdf'
        print(graph.render(filename=os.path.join("logs", 'generator.gv')))
        #np.save('../image.npy',x['image'].detach().numpy())    
        #vutils.save_image(img_proj.data,
        #                          '%s/proj_images_epoch_%03d.jpeg' % (
        #                              'output',
        #                              0,
        #                          ), normalize=True)
        break 
    

if __name__ == '__main__':


    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('--jsonfile-pkl',type=str,default ='../data/pix3d/pix3d.json',
                        help='path of the jsonfile')
    parser.add_argument('--parent-dir',type =str,default = '../data/pix3d',
                        help ='path of data file')
    parser.add_argument('--image-size', type = int, default =128,
                        help ='image size ')
    parser.add_argument('--shuffle-point-order',type=str, default= 'no',
                         help ='whether/how to shuffle point order (no/offline/online)')
    parser.add_argument('--batch-size',type=int, default= 1,
                        help='training batch size')
    parser.add_argument('--test-batch-size',type=int,default=1,
                        help='testing batch size')
    args = parser.parse_args(sys.argv[1:])
    args.cuda = torch.cuda.is_available()
    print(str(args))
    main(args)
    

    
    
  





