"""
This file holds the trainer for the segmentation network.
"""

import argparse
import numpy as np
import os
import platform

# pytorch includes
import torch
import torch.nn as nn
import torchvision as tv
from torch.autograd import Variable
from torch.utils.data import DataLoader

# ml_utils includes
from general.utils import create_session_dir, retain_session_dir

# local includes
from datasets import RGBPatches
from cropunet import CropUNet
from utils import transform_cdl

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")
if platform.node() == "matt-XPS-8900":
    DATA = HOME
else:
    DATA = "/media/data"


def main(args):
    output_dir = create_session_dir(args.output_supdir)

    dataset = RGBPatches(args.data_dir_or_file, args.labels_dir_or_file,
            mode="test")
    num_classes = dataset.get_num_cats()
#    loader = DataLoader(dataset,
#            batch_size=args.batch_size,
#            num_workers=8,
#            shuffle=True)
    model = CropUNet(num_classes=num_classes)
    model.load_state_dict( torch.load(args.model_path) )
    if args.use_cuda:
        model = model.cuda()
    model.eval()
    for i in range(len(dataset)):
        patch,label = dataset[i]
        patch.unsqueeze_(0)
        label,_ = transform_cdl(label.cpu().data.numpy())
        label = np.transpose(label, (2,0,1))
        label = torch.FloatTensor(label)
        if args.use_cuda:
            patch = Variable(patch).cuda()
            label = Variable(label).cuda()
        else:
            patch = Variable(patch)
            label = Variable(label)
        yhat = model(patch)
        preds = torch.argmax(yhat, dim=1).squeeze_()
        preds,_ = transform_cdl(preds.cpu().data.numpy())
        preds = np.transpose(preds, (2,0,1))
        preds = torch.FloatTensor(preds)
        if args.use_cuda:
            preds = Variable(preds).cuda()
        else:
            preds = Variable(preds)
        tv.utils.save_image([preds, label], pj(output_dir,
            "segments_%03d.png" % (i)))

    retain_session_dir(output_dir)


def _test_main(args):
    print("This function does not currently have any tests.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
            help="If set, just run the test code")
    parser.add_argument("--mp", "--model-path", dest="model_path", type=str,
            default=pj(HOME, "Training/cropnet/models/seg_model.pkl"))
    parser.add_argument("-o", "--output-supdir", type=str, 
            default=pj(HOME, "Training/cropnet/sessions"))
    parser.add_argument("-d", "--data-dir-or-file", type=str,
            default=pj(HOME, "Training/cropnet/sessions/session_01/feats.npy"))
    parser.add_argument("-l", "--labels-dir-or-file", type=str,
            default=pj(DATA, "Datasets/HLS/test_imgs/cdl/" \
                    "cdl_2016_neAR_0_0_500_500.npy"))
    parser.add_argument("--no-cuda", dest="use_cuda", action="store_false")
    args = parser.parse_args()
    if args.test:
        _test_main(args)
    else:
        main(args)
