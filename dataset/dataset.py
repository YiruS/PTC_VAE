'''
PointCloudDataset
author : Yefan
created : 8/9/19 10:21PM
'''
from scipy.misc import imshow
import os
import sys
import argparse
import glog as logger
import cv2
from torch.utils.data import Dataset
import json
import numpy as np
import torch
#import show3d
from torchvision import transforms, utils

import skimage.transform
import skimage.io as io

BASE_DIR = 'data/pix3d'





class PointCloudDataset(Dataset):

    def __init__(self, jsonfile, shuffle_point_order='no', image_size = 128, category_count = 9):
        '''
        Args:
             jsonfile(string): Path to the csv file with annotations.
        '''
        self.shuffle_point_order = shuffle_point_order
        self.json = open(jsonfile, "r")
        self.datalist = json.load(self.json)
        self.image_size = image_size
        self.transforms = ToTensor()
        self.oneHot = OneHot(category_count)


    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):

        img_path = os.path.join(BASE_DIR, self.datalist[idx]["img"])
        image = cv2.imread(img_path)

        mask_path = os.path.join(BASE_DIR, self.datalist[idx]["mask"])
        mask = cv2.imread(mask_path)

        ptcloud_path = os.path.join(BASE_DIR, self.datalist[idx]["ptcloud"])
        ptcloud = np.load(ptcloud_path)

        proMatrix = np.asarray(self.datalist[idx]['projection'])
        category = self.oneHot(self.datalist[idx]['category'])

        # Resize image
        convertor  = transforms.ToPILImage()
        if isinstance(image, np.ndarray) and isinstance(mask,np.ndarray):
            image = convertor(image)
            mask = convertor(mask)

        image = transforms.functional.resize(image, [self.image_size,self.image_size])
        mask = transforms.functional.resize(mask, [self.image_size,self.image_size])

        # Numpy to Tensor
        image = np.array(image) / 255.0
        mask = np.array(mask) / 255.0

        image = image.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))

        sample = {'image': image, 'ptcloud':ptcloud,\
                  'proMatrix':proMatrix,\
                   'mask':mask,
                   'category':category
                  }


        #print(sample['image'].shape)
        sample = self.transforms(sample)

        return sample

class PointCloudDataset_Cached(Dataset):

    def __init__(self, jsonfile, shuffle_point_order='no', image_size = 128,category_count = 9):
        '''
        Args:
             jsonfile(string): Path to the csv file with annotations.
        '''
        self.shuffle_point_order = shuffle_point_order
        self.json = open(jsonfile, "r")
        self.datalist = json.load(self.json)
        self.image_size = image_size
        self.transforms = ToTensor()

        self.cached_data_img = np.zeros((len(self.datalist), 3,  self.image_size, self.image_size))
        self.cached_data_mask = np.zeros((len(self.datalist), 3,  self.image_size, self.image_size))
        self.cached_data_transform = np.zeros((len(self.datalist), 4, 4))
        self.cached_data_pt = np.zeros((len(self.datalist), 1024, 3))
        self.cached_data_category = torch.zeros([len(self.datalist),category_count])
        self.oneHot = OneHot(category_count)

        for idx in range(len(self.datalist)):

            img_path = os.path.join(BASE_DIR, self.datalist[idx]["img"])
            image = cv2.imread(img_path)

            mask_path = os.path.join(BASE_DIR, self.datalist[idx]["mask"])
            mask = cv2.imread(mask_path)

            ptcloud_path = os.path.join(BASE_DIR, self.datalist[idx]["ptcloud_n"])
            ptcloud = np.asarray(np.load(ptcloud_path))

            proMatrix = np.asarray(self.datalist[idx]['projection'])
            category = self.oneHot(self.datalist[idx]['category'])

            # Resize image
            convertor  = transforms.ToPILImage()
            if isinstance(image, np.ndarray) and isinstance(mask,np.ndarray):
                image = convertor(image)
                mask = convertor(mask)
            image = transforms.functional.resize(image, [self.image_size,self.image_size])
            mask = transforms.functional.resize(mask,[self.image_size,self.image_size])

            image = np.array(image) / 255.0
            mask = np.array(mask)/ 255.0
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            image = image.transpose((2, 0, 1))
            mask = mask.transpose((2, 0, 1))
            self.cached_data_mask[idx] = mask
            self.cached_data_img[idx] = image
            self.cached_data_transform[idx] = proMatrix
            self.cached_data_pt[idx] = ptcloud
            self.cached_data_category[idx] = category
            #print(idx)
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):

        image = self.cached_data_img[idx]
        proMatrix = self.cached_data_transform[idx]
        ptcloud = self.cached_data_pt[idx]
        mask = self.cached_data_mask[idx]
        category = self.cached_data_category[idx]
        sample = {'image': image,'ptcloud':ptcloud,\
                      'proMatrix':proMatrix,\
                      'mask':mask,\
                      'category':category
                      }

        #print(sample['image'].shape)
        sample = self.transforms(sample)

        return sample


class OneHot(object):

    def __init__(self, class_count):
        '''
        pix3d: class_count: 9
        Parameters
        ----------
        class_count: int
            total amount of category
        '''
        self.metric = torch.eye(class_count)
    def __call__(self, category):
        '''
        pix3d mapping table :
        bed: 0
        bookcase:1
        chair:2
        desk: 3
        misc:4
        sofa:5
        table:6
        tool:7
        wardrobe:8

        Parameters:
        ----------
        category : string
            mark the category of instance
        n : int
            max category integer index of this dataset
            pix3d: 9
        Returns:
        -------
        binary category 1xn :obj:`int tensor`
        '''

        if category == 'bed':
            return self.metric[0,:]
        elif category == 'bookcase':
            return self.metric[1,:]
        elif category == 'chair':
            return self.metric[2,:]
        elif category == 'desk':
            return self.metric[3,:]
        elif category == 'misc':
            return self.metric[4,:]
        elif category == 'sofa':
            return self.metric[5,:]
        elif category == 'table':
            return self.metric[6,:]
        elif category == 'tool':
            return self.metric[7,:]
        elif category == 'wardrobe':
            return self.metric[8,:]





class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image,ptcloud = sample['image'], sample['ptcloud']
        proMatrix = sample['proMatrix']
        mask = sample['mask']
        category = sample['category']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = image.transpose((2, 0, 1))
        #mask = mask.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'ptcloud': torch.from_numpy(ptcloud).float(),
                'proMatrix':torch.from_numpy(proMatrix).float(),
                'mask':torch.from_numpy(mask).float(),
                'category':category.float()
                }

def main(args):
    kwargs = {'num_workers':4, 'pin_memory':True} if args.cuda else {}


    dataloader = torch.utils.data.DataLoader(
                        PointCloudDataset_Cached(args.jsonfile_pkl),
                        batch_size=args.batch_size, shuffle=True,**kwargs)
    count = 0;

    for batch in dataloader:
        print(batch["category"].shape)
        count += 1
        if count == 2:
            break


    #show3d.showpoints(dataloader[11]["ptcloud"].numpy())

if __name__ == '__main__':

    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('--jsonfile-pkl',type=str,default='../data/pix3d/minitest_pix3d.json',
                        help='path of the jsonfile')
    parser.add_argument('--image-size', type = int, default =128,
                        help ='image size ')
    parser.add_argument('--shuffle-point-order',type=str, default= 'no',
                         help ='whether/how to shuffle point order (no/offline/online)')
    parser.add_argument('--batch-size',type=int, default=8,
                        help='training batch size')
    parser.add_argument('--test-batch-size',type=int,default=8,
                        help='testing batch size')
    args = parser.parse_args(sys.argv[1:])
    args.cuda = torch.cuda.is_available()
    print(str(args))

    main(args)

