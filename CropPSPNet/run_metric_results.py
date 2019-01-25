import os
import numpy as np
import sys
import glob

from ptsemseg.metrics import runningScore, averageMeter


areaname = sys.argv[1]
gtpaths = glob.glob('{}_results_val/gt_*'.format(areaname))

metrics_val = runningScore(3)

for gtpth in gtpaths:
    outpth = gtpth.replace('gt_', 'output_')
    print('gtpath: {}'.format(gtpth))
    print('outpath: {}'.format(outpth))

    gt = np.load(gtpth)[0]
    output = np.load(outpth)[0]
    pred = np.argmax(output, axis=0)

    metrics_val.update(gt, pred)

    print('gt.size: {},  output.size: {}, pred.size: {}'.format(gt.shape, output.shape, pred.shape))

scores, class_ious = metrics_val.get_scores()
for k, v in scores.items():
    print('{}: {}'.format(k, v))

for k, v in class_ious.items():
    print('{}: {}'.format(k, v))

print('IoU: {}'.format(scores['Mean IoU : \t']))

