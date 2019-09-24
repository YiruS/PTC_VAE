#!/usr/bin/python
'''
plot_log in DeepGeom

author  : cfeng
created : 1/31/18 7:49 AM
'''

import os
import sys
import glob
import time
import argparse

import numpy as np
from matplotlib import pyplot as plt

from utils import TrainTestMonitor as TTMon
#from Tkinter import Tk
#import tkFileDialog

def main(args):
    '''
    if not os.path.exists(args.log_dir):
        tkroot = Tk()
        tkroot.withdraw()
        args.log_dir = tkFileDialog.askdirectory(title='select log folder', initialdir='../logs', mustexist=True)
        tkroot.destroy()
    assert(os.path.exists(args.log_dir))
    ttm = TTMon(args.log_dir,plot_extra=args.plot_extra!=0)
    plt.show()
    '''
    test = np.load(args.log_dir + 'iter_loss.npy')
    train_running = np.load(args.log_dir + 'stats_train_running.npz')
    plt.plot(test["iter_loss"][:,1])
    plt.show()
    plt.plot(train_running["iter_loss"][:,1])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0])

    parser.add_argument('-d','--log_dir',type=str, default='', help='log folder')
    parser.add_argument('-e','--plot_extra',type=int, default=0, help='plot training accuracy and test loss')

    args = parser.parse_args(sys.argv[1:])
    args.script_folder = os.path.dirname(os.path.abspath(__file__))

    main(args)
