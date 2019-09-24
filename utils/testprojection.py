
import json
import numpy as np
import cv2
import argparse
import os
import sys
import skimage.transform
from torchvision import transforms
from sklearn.mixture import GaussianMixture




h = np.load("h.npy").astype(int)
w = np.load("w.npy").astype(int)
image = np.load("image.npy")

print(h)
print(h.shape)
print(w.shape)
print(image.shape)
image = np.transpose(image,(0,2,3,1))
B = image.shape[0]
num = h.shape[1]

for i in range(B):
    image[i, h[i], w[i]] = [0 , 255 , 0]


for i in range(B):
    cv2.imshow("color",image[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
