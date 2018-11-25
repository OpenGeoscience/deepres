"""
This file holds the trainer for the segmentation network.
"""

import argparse
import os

# pytorch includes
import torch
import torchvision as tv

# local includes
from datasets import RGBPatches
from seg_model import CropSeg

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


def main(args):
    dataset = RGBPatches(args.data_dir_or_file, args.labels_dir_or_file,
            mode="train")
    model = CropSeg(256, args.num_classes)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-model-path", type=str, 
            default=pj(HOME, "Training/cropnet/sessions/session_10/models/" \
                    "pretrained.pkl"))
    parser.add_argument("-d", "--data-dir-or-file", type=str,
            default=pj(HOME, "Training/cropnet/sessions/session_07/feats.npy"))
    parser.add_argument("-l", "--labels-dir-or-file", type=str,
            default="/media/data/Datasets/HLS/test_imgs/cdl/" \
                    "cdl_2016_neAR_0_0_500_500.npy")
    parser.add_argument("-s", "--image-size", type=int, default=256)
    parser.add_argument("--nc", "--num-classes", dest="num_classes", type=int,
            default=4)
    parser.add_argument("--no-cuda", dest="use_cuda", action="store_false")
    args = parser.parse_args()
    main(args)
