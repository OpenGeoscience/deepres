import numpy as np
import glob
import cv2
from ptsemseg.metrics import runningScore, averageMeter

import math 
import os
import sys
from matplotlib import pyplot as plt

import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F


class tempnet(nn.Module):
    def __init__(self, in_channels, out_channels, din, temp_ksz):
        super(tempnet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.tempconv = nn.Conv3d(self.in_channels, 1, kernel_size=(temp_ksz, 1, 1), stride=1, padding=(1,0,0))
        
        self.dout = int(math.floor((din + 2 - (temp_ksz - 1) -1) + 1))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.0)
        self.classification = nn.Conv2d(self.dout, self.out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.tempconv(x)
        #print('x1.size: {}'.format(x.size()))
        x = x.view(-1, self.dout, 336, 336)
        #print('x2.size: {}'.format(x.size()))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.classification(x)

        return x

def ReadInputData(tmp_pths, foldname, areaname):
    tmp_data = []
    tmp_grd = []

    for pth in tmp_pths:
        gt = np.load(areaname + '_results_' + foldname + '/gt_02' + pth) 

        tmp_grd.append(gt[0])
        
        dat = []
        for k in range(2, 25, 1):
            if k < 10:
                output = np.load(areaname + '_results_' + foldname + '/output_0{}'.format(k) + pth)
            else:
                output = np.load(areaname + '_results_' + foldname + '/output_{}'.format(k) + pth)

            dat.append(output[0].transpose(1,2,0))
        
        #print('dat.shape: {}'.format(dat.shape))
        dat = np.array(dat).transpose(3, 0, 1, 2)
        #dat = dat.transpose(2, 0, 1)
        tmp_data.append(dat)

    return np.array(tmp_data), np.array(tmp_grd)


areaname = sys.argv[1]
tempK = int(sys.argv[2])

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

catnumfname = './configs/{}_category_number.txt'.format(areaname)
if os.path.exists(catnumfname):
    counts = np.loadtxt(catnumfname)
    nbackground = counts[0]
    ncorn = counts[1]
    nsoybean = counts[2]
else:
    k = 0
    for pthname in train_gtnames:
        k += 1
        print('k = {}, gtpathname: {}'.format(k, pthname)) 
        gt = np.load(pthname)
        nbackground += (gt == 0).sum()
        ncorn += (gt == 1).sum()
        nsoybean += (gt == 2).sum()

    catfile = open(catnumfname, 'wt')
    catfile.write('{} {} {}'.format(nbackground, ncorn, nsoybean))
    catfile.close()

# set up weights
print('nbackground = {}'.format(nbackground))
print('ncorn = {}'.format(ncorn))
print('nsoybean = {}'.format(nsoybean))

wgts = [1.0, 1.0*nbackground/ncorn, 1.0*nbackground/nsoybean]
total_wgts = sum(wgts)
wgt_background = wgts[0]/total_wgts
wgt_corn = wgts[1]/total_wgts
wgt_soybean = wgts[2]/total_wgts
weights = Variable(torch.FloatTensor([wgt_background, wgt_corn, wgt_soybean]), requires_grad=False).cuda()

# set up model
model = tempnet(3, 3, 23, tempK).cuda()
model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

test_batch_size = 1
modelpath = 'final_results/{}_best_temp_merge_model.pkl'.format(areaname)
checkpoint = torch.load(modelpath)
model.load_state_dict(checkpoint['model_state'])

model.eval()

figpath = 'final_results/{}_visualizations/'.format(areaname)
os.makedirs(figpath, exist_ok=True)

# Evaluation.
#
running_metrics_val = runningScore(3)
for step_val in range(1, ntest+1, 1):
    tmp_pths = []
    for k in range((step_val-1)*test_batch_size, step_val*test_batch_size, 1):
        kt = k % ntest
        tmp_pths.append(test_patches[kt])

    dat_val, gt_val = ReadInputData(tmp_pths, 'val', areaname)

    with torch.no_grad():
        val_dat = Variable(torch.FloatTensor(dat_val)).cuda()
        val_truth = Variable(torch.LongTensor(gt_val)).cuda()

        val_output = model(val_dat)

        val_loss = criterion(val_output, val_truth)
        
        val_pred = val_output.data.max(1)[1].cpu().numpy()[0]
        val_gt = val_truth.data.cpu().numpy()[0]

        print('pred.size: {}, gt.size: {}'.format(val_pred.shape, val_gt.shape))

        fig, (ax1, ax2) = plt.subplots(figsize=(13, 5), ncols=2)
        im1 = ax1.imshow(val_pred, cmap=plt.cm.jet, interpolation='nearest')
        fig.colorbar(im1, ax=ax1)
        ax1.set_title('Prediction')
        ax1.set_axis_off()
        ax1.set_aspect('auto')

        im2 = ax2.imshow(val_gt, cmap=plt.cm.jet, interpolation='nearest')
        fig.colorbar(im2, ax=ax2)
        ax2.set_title('Ground-truth')
        ax2.set_axis_off()
        ax2.set_aspect('auto')

        #plt.close()
        imgname = figpath + 'pred_vs_gt' + tmp_pths[0].replace('.npy', '.png')
        print('Image save path: {}'.format(imgname))

        fig.savefig(imgname, bbox_inches='tight')

        running_metrics_val.update(val_gt, val_pred)

print('testing loss = {}'.format(val_loss))
score, class_iou = running_metrics_val.get_scores()
print('output score:')
for k, v in score.items():
    print('{}: {}'.format(k, v))

print('output class_iou:')
for k, v in class_iou.items():
    print('{}: {}'.format(k, v))

