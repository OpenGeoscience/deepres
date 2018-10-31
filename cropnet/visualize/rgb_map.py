"""
This uses a pre-existing model to create a false color map using the results 
from the autoencoder.  This false color image is then to be fed into a 
traditional image segmentation routine
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import shutil
import torch

from collections import OrderedDict

# pytorch imports
from torch.utils.data import DataLoader

# ml_utils imports

# project imports
from cropnet.datasets import TBChips
from cropnet.ae_model import load_ae_model
from location_image import get_cdl_chip
from utils import get_chip_bbox, get_features, get_bbox_from_file_name, \
        get_cdl_subregion, make_clut, transform_cdl

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


g_red_idx = 3
g_green_idx = 2
g_blue_idx = 1
g_time_idx = 11

def get_data_loader(file_name, bbox):
    sz = bbox[2] - bbox[0]
    print("Getting HLS data...")
    tb_chips = TBChips(file_name, src_image_x=bbox[0], src_image_y=bbox[1],
            src_image_size=sz)
    print("...Done")
    data_loader = DataLoader(dataset=tb_chips,
            batch_size=64,
            shuffle=False,
            num_workers=8)
    return data_loader,tb_chips


def get_rgb(tb_chips):
    data = tb_chips.get_data()
    rgb = data[ [g_red_idx,g_green_idx,g_blue_idx], g_time_idx, :, : ]
    rgb = np.transpose(rgb, (1,2,0))
    return rgb

def normalize_feats(features):
    if len(features.shape) != 3:
        raise RuntimeError("Expecting features input to have 3 dimensions")
    for k in range(features.shape[2]):
        c = features[:,:,k]
        features[:,:,k] = (c - np.min(c)) / (np.max(c) - np.min(c))
    return features

def write_cats_and_features(rgb, features, cdl, cat_dict):
    nrows = 4
    ncols = 14
    grid_sz = (nrows, ncols)
    plt.figure(figsize=(28,8))

    gridspec.GridSpec(nrows, ncols)
    ax1 = plt.subplot2grid(grid_sz, (0,0), rowspan=nrows, colspan=nrows)
    ax1.imshow(rgb)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.autoscale("off")
    ax1.set_title("RGB, time index 18")


    ax2 = plt.subplot2grid(grid_sz, (0,5), rowspan=nrows, colspan=nrows)
    ax2.imshow(features)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.autoscale("off")
    ax2.set_title("AE features")

    ax3 = plt.subplot2grid(grid_sz, (0,10), rowspan=nrows, colspan=nrows)
    for k in cat_dict.keys():
        rgb = np.array(cat_dict[k]["rgb"]) / 256.0
        name = cat_dict[k]["name"]
        plt.scatter([10], [10], c=np.array([rgb]), label=name)
    ax3.imshow(cdl)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.autoscale("off")
    ax3.legend()
    ax3.set_title("CDL categories")

    plt.savefig( pj(HOME, "Training/cropnet/test_samples/features.png") )


def main(args):
    bbox = [args.subregion_x, args.subregion_y,
            args.subregion_x + args.subregion_size,
            args.subregion_y + args.subregion_size]
    data_loader,tb_chips = get_data_loader(args.data_npy_file, bbox)
    model = load_ae_model(args.model_path, args.model_name, chip_size=19,
            bneck_size=3)

    rgb = get_rgb(tb_chips)
    features = get_features(model, data_loader, bbox)
    features = normalize_feats(features)
    cdl = get_cdl_chip(args.cdl_file_path, bbox)
    cdl,cat_dict = transform_cdl(cdl)
    write_cats_and_features(rgb, features, cdl, cat_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-npy-file", type=str,
            default=pj(HOME, "Datasets/HLS/test_imgs/hls/" \
                    "hls_tb_ark_0_0_500_500.npy"))
    parser.add_argument("--cdl-file-path", type=str, 
            default=pj(HOME, "Datasets/HLS/test_imgs/cdl/" \
                    "cdl_2016_neAR_0_0_500_500.npy"))
    parser.add_argument("--model-path", type=str, 
            default=pj(HOME, "Training/cropnet/sessions/session_07/models/" \
                    "model.pkl"))
    parser.add_argument("--model-name", type=str, default="CropNetFCAE")

    parser.add_argument("--subregion-x", type=int, default=0)
    parser.add_argument("--subregion-y", type=int, default=0)
    parser.add_argument("--subregion-size", type=int, default=20)

    args = parser.parse_args()
    main(args)

