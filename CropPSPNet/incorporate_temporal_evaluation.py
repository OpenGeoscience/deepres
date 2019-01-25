import numpy as np
import glob
import cv2
from ptsemseg.metrics import runningScore, averageMeter

import math 
import os
import sys

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

batch_size = 40
test_batch_size = 1

start_step = 1
num_steps = 500

best_iou = 0

for step in range(start_step, num_steps, 1):
    tmp_pths = []
    for k in range((step-1)*batch_size, step*batch_size, 1):
        kt = k % ntrain
        tmp_pths.append(train_patches[kt])

    dat, gt = ReadInputData(tmp_pths, 'train', areaname)
    print('step: {}'.format(step))
    print('dat.size: {}'.format(dat.shape))
    print('gt.size: {}'.format(gt.shape))

    model.train()
    optimizer.zero_grad()

    with torch.no_grad():
        var_dat = Variable(torch.FloatTensor(dat)).cuda()
        var_gt = Variable(torch.LongTensor(gt)).cuda()

    output = model(var_dat)
    print('output.size: {}'.format(output.size()))
    loss = criterion(output, var_gt)

    pred = output.data.max(1)[1].cpu().numpy()
    gt = var_gt.data.cpu().numpy()

    # validation
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
            
            val_pred = val_output.data.max(1)[1].cpu().numpy()
            val_gt = val_truth.data.cpu().numpy()

            running_metrics_val.update(val_gt, val_pred)

    print('training loss = {}'.format(loss))
    print('testing loss = {}'.format(val_loss))
    score, class_iou = running_metrics_val.get_scores()
    print('output score:')
    for k, v in score.items():
        print('{}: {}'.format(k, v))

    print('output class_iou:')
    for k, v in class_iou.items():
        print('{}: {}'.format(k, v))
    
    meanIoU = score['Mean IoU : \t']
    if meanIoU > best_iou:
        best_iou = meanIoU
        resfname = 'final_results/{}_best_result.txt'.format(areaname)
        resfile = open(resfname, 'wt')

        for k, v in score.items():
            resfile.write('{}: {}\n'.format(k, v))

        print('output class_iou:')
        for k, v in class_iou.items():
            resfile.write('{}: {}\n'.format(k, v))

        resfile.close()
        
        state = {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_iou": best_iou,
        }

        save_path = 'final_results/{}_best_temp_merge_model.pkl'.format(areaname)
        
        torch.save(state, save_path)
        
    
    loss.backward()
    optimizer.step()


