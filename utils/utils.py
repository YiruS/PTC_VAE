import os
import sys
import argparse
import errno

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, proj3d

import torch
from torch.nn import init


def make_D_label(input, value, device, random=False):
	if random:
		if value == 0:
			lower, upper = 0, 0.205
		elif value ==1:
			lower, upper = 0.8, 1.05
		D_label = torch.FloatTensor(input.data.size()).uniform_(lower, upper).to(device)
	else:
		D_label = torch.FloatTensor(input.data.size()).fill_(value).to(device)

	return D_label


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
	net.apply(init_func)  # apply the initialization function <init_funce

def check_exist_or_mkdirs(path):
    '''thread-safe mkdirs if not exist'''
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def vis_pts(pts, clr, cmap):
    fig = plt.figure()
    fig.set_rasterized(True)
    ax = axes3d.Axes3D(fig)

    ax.set_alpha(0)
    ax.set_aspect('equal')
    min_lim = pts.min()
    max_lim = pts.max()
    ax.set_xlim3d(min_lim,max_lim)
    ax.set_ylim3d(min_lim,max_lim)
    ax.set_zlim3d(min_lim,max_lim)

    if clr is None:
        M = ax.get_proj()
        _,_,clr = proj3d.proj_transform(pts[:,0], pts[:,1], pts[:,2], M)
        clr = (clr-clr.min())/(clr.max()-clr.min())

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    ax.scatter(
        pts[:,0],pts[:,1],pts[:,2],
        c=clr,
        zdir='x',
        s=20,
        cmap=cmap,
        edgecolors='k'
    )
    return fig


def count_parameter_num(params):
    cnt = 0
    for p in params:
        cnt += np.prod(p.size())
    return cnt


class TrainTestMonitor(object):

    def __init__(self, log_dir, plot_loss_max=4., plot_extra=False):
        assert(os.path.exists(log_dir))

        stats_test = np.load(os.path.join(log_dir, 'stats_test.npz'))
        stats_train_running = np.load(os.path.join(log_dir, 'stats_train_running.npz'))

        self.title = os.path.basename(log_dir)
        self.fig, self.ax1 = plt.subplots()
        self.ax2 = self.ax1.twinx()
        plt.title(self.title)

        # Training loss
        iter_loss = stats_train_running['iter_loss']
        self.ax1.plot(iter_loss[:,0], iter_loss[:,1],'-',label='train loss',color='r',linewidth=2)
        self.ax1.set_ylim([0, plot_loss_max])
        self.ax1.set_xlabel('iteration')
        self.ax1.set_ylabel('loss')

        # Test accuracy
        iter_acc = stats_test['iter_acc']
        max_accu_pos = np.argmax(iter_acc[:,1])
        test_label = 'max test accuracy {:.3f} @ {}'.format(iter_acc[max_accu_pos,1],max_accu_pos+1)
        self.ax2.plot(iter_acc[:,0], iter_acc[:,1],'o--',label=test_label,color='b',linewidth=2)
        self.ax2.set_ylabel('accuracy')

        if plot_extra:
            # Training accuracy
            iter_acc = stats_train_running['iter_acc']
            self.ax2.plot(iter_acc[:,0], iter_acc[:,1],'--',label='train accuracy',color='b',linewidth=.8)
            # Test loss
            iter_loss = stats_test['iter_loss']
            self.ax1.plot(iter_loss[:,0], iter_loss[:,1],'--',label='test loss',color='r',linewidth=.8)

        self.ax1.legend(loc='upper left', framealpha=0.8)
        self.ax2.legend(loc='lower right', framealpha=0.8)
        self.fig.show()


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0])

    args = parser.parse_args(sys.argv[1:])
    args.script_folder = os.path.dirname(os.path.abspath(__file__))

    main(args)
