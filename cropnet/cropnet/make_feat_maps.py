"""
Take an AE model and run inference to generate full feature maps for each of 
the 4 regions
"""

import argparse
import csv
import logging
import numpy as np
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

# pytorch imports
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# ml_utils imports
from general.utils import create_session_dir, retain_session_dir

# Local imports
from ae_model import CropNetCAE, CropNetFCAE, load_ae_model
from ae_trainer import AETrainer
from datasets import RGBPatchesCenter, TBChips
from utils import get_features

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


g_session_log = "session.log"


def main(args):
    model = get_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="CropNetCAE",
            choices=["CropNetCAE", "CropNetFCAE"])
    parser.add_argument("-d", "--data-dir", type=str,
            default=pj(HOME, "Datasets/HLS/tb_data/all/hls"))
    args = parser.parse_args()
    main(args)

