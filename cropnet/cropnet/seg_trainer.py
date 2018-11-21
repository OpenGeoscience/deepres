"""
This file holds the trainer for the segmentation network.
"""

import argparse
import os

# pytorch includes
import torch
import torchvision as tv
from torchvision.models import resnet18

# local includes
from datasets import RGBPatches
from unet import *

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


def main(args):
    encoder = resnet18(pretrained=True)
    dyunet = DynamicUnet(encoder, n_classes=args.num_classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-model-path", type=str, 
            default=pj(HOME, "Training/cropnet/sessions/session_10/models/" \
                    "pretrained.pkl"))
    parser.add_argument("--nc", "--num-classes", dest="num_classes", type=int,
            default=4)
    args = parser.parse_args()
    main(args)
