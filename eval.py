
import os
import sys
import argparse
import glog as logger
import torch
import torch.utils.data
import torch.nn as nn
import torch.autograd
import optparse
import time
from dataset.dataset import PointCloudDataset_Cached,PointCloudDataset
from model.ProGenerator import GeneratorVanilla
from utils.utils import count_parameter_num
from Protraintester import ChamfersDistance,TrainTester
from torch.utils.tensorboard import SummaryWriter



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


def main(args):
        # load data

    writer = SummaryWriter()
    kwargs = {'num_workers':4, 'pin_memory':True}
    
    test_loader = torch.utils.data.DataLoader(
                        PointCloudDataset(args.test_json),
                        batch_size=args.test_batch_size, shuffle= True,**kwargs)
    

    netG = GeneratorVanilla(
           encoder_dim = (3,3),
           grid_dims=(32,32,1),
           Generate1_dims=(1219,128,128,3),
           Generate2_dims=(1219,256,256,3))


    netG.load_state_dict(torch.load('snapshots/model_train_best.pth'))

    netG.eval()

    criterion_I = torch.nn.L1Loss().to(args.device)
    criterion_G = ChamfersDistance().to(args.device)
    

    for batch in test_loader:
            image, ptcloud = batch['image'], batch['ptcloud']
            proMatrix = batch['proMatrix']
            
            image, ptcloud = Variable(image).to(args.device), \
                             Variable(ptcloud).to(args.device)
           
            proMatrix = Variable(proMatrix,requires_grad=False).to(args.device)

                                                                                
            B = image.shape[0]
            
            noise = cube_generator(B,args.noise_pts,3)
            noisev = Variable(noise).to(args.device)
            
                 
            
            with torch.set_grad_enabled(False):
                h, w, img_recons, ptcloud_pred_primitive, ptcloud_pred_fine = netG(image,noisev,proMatrix)

            colors_tensor = torch.zeros(B, args.noise_pts, 3)

            writer.add_mesh('pt_fine', vertices=ptcloud_pred_fine, colors=colors_tensor)
            writer.close()

if __name__ == "__main__":
    parser = optparse.OptionParser(sys.argv[0], description="Eval")

    parser.add_option("--test-json",
                      dest="test_json", type=str,
                      default="data/pix3d/testnet.json",
                      help='path of the testing json file')

    parser.add_option("--test-batch-size",
                      dest="test_batch_size", type=int,
                      default=1,
                    help='testing batch size') 

    parser.add_option("--noise-pts",
                      dest="noise_pts", type=int, 
                      default=1024,
                      help="#points in grid")

    (args, opts) = parser.parse_args()
   
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.script_folder = os.path.dirname(os.path.abspath(__file__))
    print(str(args))
    sys.stdout.flush()
    main(args)