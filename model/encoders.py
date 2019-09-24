'''
Encoder(2D conv + FC layer) for 2D image

author : Yefan
created : 8/10/19 10:23 AM
'''
import os
import sys
import math
import argparse
import torch

from torch import nn
from torch.nn import init
from torch.autograd import Variable
from torchvision import models
from torch.nn import functional as F
#from dataset import PointCloudDataset

from graphviz import Digraph
from torchviz import make_dot

from deeplab import ResNet, Bottleneck
from SegNet import SegNet_Small, SegNet

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/.." % file_path)

from utils.torchsummary import summary

def init_weights(net, init_type="kaiming", init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def conv2d_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, stride=2, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )

def linear_block(in_c, out_c):
    return nn.Sequential(
        nn.Linear(in_c, out_c),
        nn.BatchNorm1d(out_c),
        nn.ReLU(),
    )

class Encoder(nn.Module):
    """Encoder of Structure Generator"""
    def __init__(self, Conv2d_dims,FC_dims):
        super(Encoder, self).__init__()
        self.FC_dims = FC_dims
        self.conv1 = conv2d_block(Conv2d_dims[0], Conv2d_dims[1])
        self.conv2 = conv2d_block(Conv2d_dims[1], Conv2d_dims[2])
        self.conv3 = conv2d_block(Conv2d_dims[2], Conv2d_dims[3])
        self.conv4 = conv2d_block(Conv2d_dims[3], Conv2d_dims[4])
        self.conv5 = conv2d_block(Conv2d_dims[4], Conv2d_dims[5])
        self.pool =  nn.MaxPool2d(2)
        self.fc1 = linear_block(FC_dims[0], FC_dims[1]) # After flatten
        self.fc2 = nn.Linear(FC_dims[1], FC_dims[2])
        
        



    def forward(self, x):
        
        x = self.conv1(x)
        
        x = self.conv2(x)
        
        x = self.conv3(x)
        
        x = self.conv4(x)
        
        x = self.conv5(x)
        
        x = self.pool(x)
        
        x = self.fc1(x.view(x.shape[0], self.FC_dims[0]))
        
        x = self.fc2(x)
        
        return x

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, out_dim):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2) # stride=2
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 128, layers[2], stride=2)  # (block, 256, , ,)
        # self.layer4 = self._make_layer(block, 128, layers[3], stride=2)  # (block, 256, , ,)
        # # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(2048, out_dim)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(8192, out_dim)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x))) # Bx64x64x64
        x = self.relu3(self.bn3(self.conv3(x))) # Bx64x64x64
        x = self.maxpool(x)  # Bx128x32x32

        x = self.layer1(x) # 8x64x32x32
        x = self.layer2(x) # 8x128x16x16
        x = self.layer3(x) # 8x128x8x8
        x = self.layer4(x) # Bx128x4x4

        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Bx2048
        x = self.fc(x)
        #
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x

def resnet18(layers, out_dim):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = ResNet(block=BasicBlock, layers=layers, out_dim = out_dim)
    init_weights(model, init_type="kaiming")
    return model


def main(args):
    #model = resnet18(
    #       layers=[2, 2, 2, 2],
    #       out_dim=3,
    #   )
    # model = ResNet(block=Bottleneck, layers=[3, 4, 23, 3], num_channels=3, image_size=[128, 128])
    # model = SegNet_Small(input_channels=3, num_classes=3, skip_type="mul")
    model = SegNet(input_channels=3, output_channels=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    init_weights(model, init_type="kaiming")
    #model.to("cuda")
    #summary(model, input_size=(3, 128, 128))

    img = Variable(torch.randn(8, 3, 128, 128))#.to("cuda")
    latent, out= model(img)
    graph = make_dot(out)
    graph.engine = 'dot'
    graph.format = 'pdf'
    print(graph.render(filename=os.path.join("logs", 'encoder.gv')))
    print(latent.shape, out.shape)







    #kwargs = {'num_workers':1, 'pin_memory':True} if args.cuda else {}
    #encoder = Encoder(Conv2d_dims=(3, 96, 128, 192, 256, 512),
    #                  FC_dims =(2048, 1024, 512))
    #dataloader = torch.utils.data.DataLoader(
    #                    PointCloudDataset(args.jsonfile_pkl, args.parent_dir, args.image_size),
    #                    batch_size=args.batch_size, shuffle= False, **kwargs)
    #for x in dataloader:
    #    encoder(x["image"])
    #    break



if __name__ == '__main__':

    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('-0','--jsonfile-pkl',type=str,
                        help='path of the jsonfile')
    parser.add_argument('-d','--parent-dir',type =str,
                        help ='path of data file')
    parser.add_argument('--image-size', type = int, default = 256,
                        help ='image size ')
    parser.add_argument('--shuffle-point-order',type=str, default= 'no',
                         help ='whether/how to shuffle point order (no/offline/online)')
    parser.add_argument('--batch-size',type=int,default=16,
                        help='training batch size')
    parser.add_argument('--test-batch-size',type=int,default=32,
                        help='testing batch size')
    args = parser.parse_args(sys.argv[1:])
    args.cuda = torch.cuda.is_available()

    main(args)
