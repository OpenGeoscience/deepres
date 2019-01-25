import numpy as np
import glob
import cv2

import math 
import os
import sys

areaname = sys.argv[1]

train_gtnames = glob.glob('{}_results_train/gt_02_*'.format(areaname))
train_patches = [pthname.split('{}_results_train/gt_02'.format(areaname))[1] for pthname in train_gtnames]
test_gtnames = glob.glob('{}_results_val/gt_02_*'.format(areaname))
test_patches = [pthname.split('{}_results_val/gt_02'.format(areaname))[1] for pthname in test_gtnames]
print('train_patches: {}'.format(train_patches))
print('test_patches: {}'.format(test_patches))

ntrain = len(train_patches)
ntest = len(test_patches)

nbackground = 0
ncorn = 0
nsoybean = 0

k = 0
for pthname in test_gtnames:
    k += 1
    print('k = {}, gtpathname: {}'.format(k, pthname)) 
    gt = np.load(pthname)
    nbackground += (gt == 0).sum()
    ncorn += (gt == 1).sum()
    nsoybean += (gt == 2).sum()


# set up weights
print('nbackground = {}'.format(nbackground))
print('ncorn = {}'.format(ncorn))
print('nsoybean = {}'.format(nsoybean))

