import numpy as np
import glob
import cv2
from ptsemseg.metrics import runningScore, averageMeter

gtnames = glob.glob('results_val/gt_02_*')
test_patches = [pthname.split('results_val/gt_02')[1] for pthname in gtnames]
print('test_patches: {}'.format(test_patches))

running_metrics_output = runningScore(3)
running_metrics_vote = runningScore(3)
#verify the gt information
for pth in test_patches:
    print('pth: {}'.format(pth))
    #gt = cv2.imread('/home/chengjjang/Projects/deepres/SatelliteData/patchark_val/gt/ark_gt' + pth.replace('npy', 'png'), 0)
    gt = np.load('results_val/gt_02' + pth)
    for k in range(2, 25, 1):
        if k < 10:
            tmp_gt = np.load('results_val/gt_0{}'.format(k) + pth)
        else:
            tmp_gt = np.load('results_val/gt_{}'.format(k) + pth)

        check_gt = sum(sum(sum(gt-tmp_gt)))
        print('k={}, check_gt: {}, size: {}'.format(k, check_gt, gt.shape))

    
    # predict based on the output
    output = np.zeros((1, 3, gt.shape[1], gt.shape[2]))
    for k in range(2, 25, 1):
        if k < 10:
            output += np.load('results_val/output_0{}'.format(k) + pth)
        else:
            output += np.load('results_val/output_{}'.format(k) + pth)
    
    predout = np.argmax(output, axis=1)
    running_metrics_output.update(gt, predout)
    
    # majority voting based on the prediction
    predcount = np.zeros((1, 3, gt.shape[1], gt.shape[2]))
    predcount0 = np.zeros((1, gt.shape[1], gt.shape[2]))
    predcount1 = np.zeros((1, gt.shape[1], gt.shape[2]))
    predcount2 = np.zeros((1, gt.shape[1], gt.shape[2]))
    for k in range(2, 25, 1):
        if k < 10:
            pred = np.load('results_val/pred_0{}'.format(k) + pth)
        else:
            pred = np.load('results_val/pred_{}'.format(k) + pth)
        
        predcount0[pred==0] += 1
        predcount1[pred==1] += 1
        predcount2[pred==2] += 1
   
    predcount[:, 0, :, :] = predcount0
    predcount[:, 1, :, :] = predcount1
    predcount[:, 2, :, :] = predcount2
    predvote = np.argmax(predcount, axis=1) 
    running_metrics_vote.update(gt, predvote)


score, class_iou = running_metrics_output.get_scores()
print('output score:')
for k, v in score.items():
    print('{}: {}'.format(k, v))

print('output class_iou:')
for k, v in class_iou.items():
    print('{}: {}'.format(k, v))
    
print('vote score:')
score, class_iou = running_metrics_vote.get_scores()
for k, v in score.items():
    print('{}: {}'.format(k, v))

print('vote class_iou:')
for k, v in class_iou.items():
    print('{}: {}'.format(k, v))
    


    

