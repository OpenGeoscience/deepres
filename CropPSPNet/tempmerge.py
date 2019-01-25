import numpy as np
import glob
import cv2
from ptsemseg.metrics import runningScore, averageMeter

import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F


class tempnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(tempnet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        #self.tempconv = nn.Conv3d(69, 69, kernel_size=(3, 1, 1), stride=1, padding=(1,1,1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.classification = nn.Conv2d(69, self.out_channels, 1, 1, 0)

    def forward(self, x):
        #x = self.tempconv(x)
        #x = x.view(-1)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.classification(x)

        return x

def ReadInputData(tmp_pths, foldname):
    tmp_data = []
    tmp_grd = []

    for pth in tmp_pths:
        gt = np.load('results_' + foldname + '/gt_02' + pth) 

        tmp_grd.append(gt[0])
        
        dat = None
        for k in range(2, 25, 1):
            if k < 10:
                output = np.load('results_' + foldname + '/output_0{}'.format(k) + pth)
            else:
                output = np.load('results_' + foldname + '/output_{}'.format(k) + pth)

            if dat is None:
                dat = output[0].transpose(1,2,0)
            else:
                dat = np.dstack((dat, output[0].transpose(1,2,0)))
        
        #print('dat.shape: {}'.format(dat.shape))
        dat = dat.transpose(2, 0, 1)
        tmp_data.append(dat)

    return np.array(tmp_data), np.array(tmp_grd)



train_gtnames = glob.glob('results_train/gt_02_*')
train_patches = [pthname.split('results_train/gt_02')[1] for pthname in train_gtnames]
test_gtnames = glob.glob('results_val/gt_02_*')
test_patches = [pthname.split('results_val/gt_02')[1] for pthname in test_gtnames]
print('train_patches: {}'.format(train_patches))
print('test_patches: {}'.format(test_patches))

ntrain = len(train_patches)
ntest = len(test_patches)


model = tempnet(69, 3).cuda()
model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

batch_size = 3

start_step = 1
num_steps = 100


for step in range(start_step, num_steps, 1):
    tmp_pths = []
    for k in range((step-1)*batch_size, step*batch_size, 1):
        kt = k % ntrain
        tmp_pths.append(train_patches[kt])

    dat, gt = ReadInputData(tmp_pths, 'train')
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
        for k in range((step_val-1)*batch_size, step_val*batch_size, 1):
            kt = k % ntest
            tmp_pths.append(test_patches[kt])

        dat_val, gt_val = ReadInputData(tmp_pths, 'val')

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
    
    
    loss.backward()
    optimizer.step()


