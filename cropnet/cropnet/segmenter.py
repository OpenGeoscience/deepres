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
from utils import transform_cdl, get_cat_list

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")
if platform.node() == "matt-XPS-8900":
    DATA = HOME
else:
    DATA = "/media/data"


def main(args):
    session_dir = os.path.dirname(args.data_dir_or_file)
    supdir = pj(session_dir, "segmentations")
    output_dir = create_session_dir(supdir, dir_stub="segment_%02d")
    cats = get_cat_list(args.cats)
    dataset = RGBPatches(data_dir=args.data_dir_or_file,
            labels_dir=args.labels_dir_or_file,
            cats=cats,
            mode="test")
    num_classes = len(cats)
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
        label,_ = transform_cdl(label.cpu().data.numpy(), cats)
        label = np.transpose(label, (2,0,1))
        label = torch.FloatTensor(label)
        if args.use_cuda:
            patch = Variable(patch).cuda()
            label = Variable(label).cuda()
        else:
            patch = Variable(patch)
            label = Variable(label)
        if patch.shape[2] != dataset.get_ingest_patch_size() or \
                patch.shape[3] != dataset.get_ingest_patch_size():
            continue
        yhat = model(patch)
#        print(patch.shape, yhat.shape)
#        print(torch.min(patch), torch.max(patch), torch.median(patch))
#        print("\t", torch.min(yhat), torch.max(yhat), torch.median(yhat))
        preds = torch.argmax(yhat, dim=1).squeeze_()
        preds,_ = transform_cdl(preds.cpu().data.numpy(), cats)
        preds = np.transpose(preds, (2,0,1))
        preds = torch.FloatTensor(preds)
        if args.use_cuda:
            preds = Variable(preds).cuda()
        else:
            preds = Variable(preds)
        patch = patch.squeeze()
        tv.utils.save_image([patch, preds, label], pj(output_dir,
            "segments_%03d.png" % (i)))

    retain_session_dir(output_dir)
    print("Wrote output maps to %s" % output_dir)


def _test_main(args):
    print("This function does not currently have any tests.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
            help="If set, just run the test code")
    parser.add_argument("--mp", "--model-path", dest="model_path", type=str,
            default=pj(HOME, "Training/cropnet/models/seg_model.pkl"))
    parser.add_argument("-d", "--data-dir-or-file", type=str,
            default=pj(HOME, "Training/cropnet/sessions/session_07/feats_maps/"\
                    "feats.npy"))
    parser.add_argument("-l", "--labels-dir-or-file", type=str,
            default=pj(DATA, "Datasets/HLS/test_imgs/cdl/" \
                    "cdl_2016_neAR_0_0_500_500.npy"))
    parser.add_argument("--no-cuda", dest="use_cuda", action="store_false")
    parser.add_argument("--cats", "--categories", dest="cats", type=str,
            default="original4", choices=["original4", "top5", "top10", 
                "top25", "top50", "all"])
    args = parser.parse_args()
    if args.test:
        _test_main(args)
    else:
        main(args)
