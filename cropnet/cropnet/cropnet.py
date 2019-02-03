"""
This file trains the cropnet models, either the AE or classifier depending on 
choice.
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

def get_data_loader(data_dir, batch_size, num_workers, resize_to=32):
    dataset = TBChips(data_dir=data_dir, resize_to=resize_to,
            tiles_per_cohort=5)
    train_loader = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return train_loader # TODO test loader!!

def make_ae_model(ae_network, chip_size, bneck_size):
    if ae_network == "CropNetFCAE":
        model = CropNetFCAE(chip_size, bneck_size)
    elif ae_network == "CropNetAE":
        model = CropNetAE(chip_size, bneck_size)
    elif ae_network == "CropNetCAE":
        model = CropNetCAE(chip_size=chip_size, bneck_size=bneck_size,
                base_nchans=16)
    else:
        raise RuntimeError("Unrecognized network %s" % (ae_network))
    return model.cuda()

def main(args):
    session_dir = create_session_dir(args.output_supdir)
    train_ae_loader = get_data_loader(args.data_dir, args.batch_size,
            resize_to=args.chip_size, num_workers=args.num_workers)
    test_ae_loader = get_data_loader(args.test_data_dir, args.batch_size,
            resize_to=args.chip_size, num_workers=args.num_workers)
    ae_model = make_ae_model(args.model, args.chip_size, args.bneck_size)
    ae_trainer = AETrainer(
            input_size=args.chip_size,
            model=ae_model,
            loaders=(train_ae_loader, test_ae_loader),
            session_dir=session_dir)
    ae_trainer.train()

    retain_session_dir(session_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-d", "--data-dir", type=str,
            default=pj(HOME, "Datasets/HLS/tb_data/train/hls"))
    parser.add_argument("--test-data-dir", type=str,
            default=pj(HOME, "Datasets/HLS/tb_data/test/hls"))
    parser.add_argument("-o", "--output-supdir", type=str,
            default=pj(HOME, "Training/cropnet/sessions"))

    parser.add_argument("--src-image-x", type=int, default=0,
            help="Source chip top coordinate")
    parser.add_argument("--src-image-y", type=int, default=0,
            help="Source chip left coordinate")
    parser.add_argument("--src-image-size", type=int, default=500)
    parser.add_argument("--model", type=str, default="CropNetCAE",
            choices=["CropNetCAE", "CropNetFCAE", "CropNetAE", "CropSeg",
                "Pretrained"])
    parser.add_argument("--chip-size", type=int, default=19)
    parser.add_argument("--bneck-size", type=int, default=3)

    parser.add_argument("--test-image-x", type=int, default=500,
            help="Test chip top coordinate")
    parser.add_argument("--test-image-y", type=int, default=0,
            help="Test chip left coordinate")
    parser.add_argument("--test-image-size", type=int, default=500)

    parser.add_argument("--opt-name", type=str, default="Adam",
            choices=["Adam", "SGD"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    
    args = parser.parse_args()
    main(args)

