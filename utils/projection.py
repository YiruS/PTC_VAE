import show3d
import json
import numpy as np
import cv2
import argparse
import os
import sys
import skimage.transform
from torchvision import transforms
from sklearn.mixture import GaussianMixture

sensor_width =  32                                      # unit mm 
import torch
import torch.nn as nn


## camera intrinc matrix convert image from sensor space to raster space 
def camera(focal, w, h):
    #focal = focal * 3
    sensor_height = sensor_width * h/ w
    matrix = np.array([[(focal*w)/sensor_width, 0, w/2 ],[0, (focal*h)/sensor_height, h/2], [0,0,1]])
    return matrix





## transformation from obj coordinate to camera coordinate
def transform(rot_mat, trans_mat):
    ## transform matrix 
    rot_mat = np.asarray(rot_mat, dtype=np.float32)
    
    
    trans_mat = np.expand_dims(np.asarray(trans_mat, dtype=np.float32),axis=1)
    print(rot_mat.shape)
    print(trans_mat.shape)
    transform = np.concatenate((rot_mat, trans_mat),axis = 1)
    
    scale = np.expand_dims(np.array([0,0,0,1]),axis=1)
    transform = np.concatenate((transform, scale.T),axis = 0)
    return transform


def load_data(args):
    jsonfile = open(args.path, "r")
    datalist = json.load(jsonfile)
    return datalist

def gaussian_generator(N, D):
	if D == 2:
    #set gaussian ceters and covariances in 2D
		means = np.array([[0.5, 0.0],
                      [0, 0],
                      [-0.5, -0.5],
                      [-0.8, 0.3]])
		covs = np.array([np.diag([0.01, 0.01]),
                     np.diag([0.025, 0.01]),
                     np.diag([0.01, 0.025]),
                     np.diag([0.01, 0.01])])
	elif D == 3:
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
	print(n_gaussians)

	points = []
	for i in range(len(means)):
		x = np.random.multivariate_normal(means[i], covs[i], N )
		points.append(x)
	points = np.concatenate(points)
	#fit the gaussian model
	gmm = GaussianMixture(n_components=n_gaussians, covariance_type='diag')
	gmm.fit(points)
	#show3d.showpoints(points)
	return points

def cube_generator(N, D):
    x_count = 8
    y_count = 8
    z_count = 16
    count = x_count * y_count * z_count
    
    cube = np.zeros((x_count*y_count*z_count, 3))
    # setting vertices
    for i in range(count):
        x = float(i % x_count) / (x_count - 1)
        y = float((i / x_count) % y_count) / (y_count - 1)
        z = float(i / (x_count * y_count) % z_count) / (z_count - 1)
        cube[i] = [x - 0.5, y - 0.5, z -0.5]
    
    cube *= 0.1/cube.max() 
    print(cube.min())
    return cube

def points_normalize(points):
    assert (points.shape[0]==1024)
    #points_mean = (np.max(points,axis=0)+np.min(points,axis=0))/2
    points_mean = np.mean(points,axis=0)
    points_shifted = points - points_mean
    max_norm =  np.max(np.linalg.norm(points_shifted,axis=1))
    #points_normalized = points_shifted/max_norm
    return points_mean, max_norm


def main(args):
	## Load data
    datadir = "../data/pix3d/"
    datalist = load_data(args)
    data = datalist[args.idex] 
    ptcloud_w_n = np.load(datadir + data['ptcloud_n'])            #ptcloud
    #ptcloud_w = np.load(datadir + data['ptcloud'])

    #points_mean, max_norm = points_normalize(ptcloud_w)
    #print('points_mean',points_mean.shape)
    #print('max_norm',max_norm)

    #revenormal = np.array([[max_norm,0,0,points_mean[0]],
    #                       [0,max_norm,0,points_mean[1]],
    #                       [0,0,max_norm,points_mean[2]],
    #                       [0,0,0,1]     ])
    #print(revenormal)
    infill = np.ones((1, ptcloud_w_n.shape[0])) 
    ptcloud_w_n = np.concatenate((ptcloud_w_n.T, infill))      
    #print(ptcloud_w_n.shape)


    image = cv2.imread(datadir + data['img'])                 #image
    convertor = transforms.ToPILImage()
    image = convertor(image)
    image = transforms.functional.resize(image,[128,128])
    image = np.array(image)
    projection  = np.asarray(data['projection'])
    #projection  = np.matmul(projection, revenormal)
    #print(projection)
    img_annoation = np.matmul(projection,ptcloud_w_n)

    img_annoation[0,:] = img_annoation[0,:]/img_annoation[2,:]
    img_annoation[1,:] = img_annoation[1,:]/img_annoation[2,:]
    img_annoation = img_annoation.astype(int)

    '''
    w = data['img_size'][0]                                   #image size
    h = data['img_size'][1]

    ## resize image to (128, 128)
    convertor = transforms.ToPILImage()
    image = convertor(image)
    image = transforms.functional.resize(image,[128,128])
    image = np.array(image)

    focal = data['focal_length']                             # focal length
    mirror_angle = np.pi                                     # mirror angle
    
    init_points = cube_generator(1024, 3)
    print(np.amax(init_points))
    ptcloud_w = init_points
    ##infill ptcloud (3, 2048) to (4, 2048)  
    infill = np.ones((1, ptcloud_w.shape[0]))                
    ptcloud_w = np.concatenate((ptcloud_w.T, infill))      
    
    
    #apply transformation from obj coordinate to camera coordinate
    trans_matrix = transform(data["rot_mat"],data["trans_mat"])
    ptcloud_c = np.matmul(trans_matrix, ptcloud_w)
    
    
    ## flip around z axis through camera spot 
    cam_matrix_z = np.array([[np.cos(mirror_angle), -np.sin(mirror_angle),0, 0], 
    	                  [np.sin(mirror_angle), np.cos(mirror_angle),0, 0],
    	                   [ 0,0,1,0], 
    	                    [0,0,0,1]               ])

    ptcloud_c = np.matmul(cam_matrix_z, ptcloud_c)
    ptcloud_c = ptcloud_c[:3, :]
    
    
    ##  camera cooridance to picture via z axis
    
    ptcloud_c[0,:] = ptcloud_c[0,:]/ptcloud_c[2,:]
    ptcloud_c[1,:] = ptcloud_c[1,:]/ptcloud_c[2,:]
    ptcloud_c[2,:] = ptcloud_c[2,:]/ptcloud_c[2,:]
    
    ##  from sensor space to raster space   unit: mm to pixel
    camera_trans = camera(focal, w, h)
    img_annoation = np.matmul(camera_trans,ptcloud_c)
    img_annoation = img_annoation.astype(int)

    ## resize img_annoation in (128, 128)
    img_annoation[1] = img_annoation[1]*128/h
    img_annoation[0] = img_annoation[0]*128/w
    img_annoation = img_annoation.astype(int)

    '''
   
    ## projection  
    ptnum_inimage = 0
    for i in range(img_annoation.shape[-1]):
        if 0 < img_annoation[1][i] < 128 and 0 < img_annoation[0][i] < 128:
            ptnum_inimage += 1
            image[img_annoation[1][i], img_annoation[0][i]] = [0 , 255 , 0]
    print("The num of ptlcoud projected to image is",ptnum_inimage)        
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('-p', '--path', type=str,
                        help='path of the jsonfile')
    parser.add_argument('-n', '--idex', type=int,
                        help='idex for all data')

    args = parser.parse_args(sys.argv[1:])

    main(args)
