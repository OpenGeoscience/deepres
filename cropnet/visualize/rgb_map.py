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
from utils import get_chip_bbox, get_features, get_bbox_from_file_path, \
        get_cdl_subregion, make_clut, transform_cdl, normalize_feats

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


g_red_idx = 3
g_green_idx = 2
g_blue_idx = 1
g_time_idx = 11

def get_data_loader(region, bbox):
    sz = bbox[2] - bbox[0]
    print("Getting HLS data...")
    data_dir = pj(HOME,"Datasets/HLS/tb_data/test/hls")
    labels_dir = pj(HOME,"Datasets/HLS/tb_data/test/cdl")
    if region=="ark":
        data_file = "hls_tb_ark_1000_1000_2000_2000.npy"
        labels_file = "cdl_2016_neAR_1000_1000_2000_2000.npy"
    elif region=="ohio":
        data_file="hls_tb_ohio_1000_1000_2000_2000.npy"
        labels_file="cdl_2016_nwOH_1000_1000_2000_2000.npy"
    elif region=="sd":
        data_file="hls_tb_sd_1000_1000_2000_2000.npy"
        labels_file="cdl_2016_seSD_1000_1000_2000_2000.npy"
    elif region=="vai":
        data_file="hls_tb_vai_1000_1000_2000_2000.npy"
        labels_file="cdl_2016_vai_crop_1000_1000_2000_2000.npy"
    else:
        raise RuntimeError("Unrecognized region, %s" % region)
    tb_chips = TBChips(data_dir=data_dir, labels_dir=labels_dir,
            data_file=data_file, labels_file=labels_file)
    print("...Done")
    data_loader = DataLoader(dataset=tb_chips,
            batch_size=64,
            shuffle=False,
            num_workers=8)
    return data_loader,tb_chips,pj(labels_dir,labels_file)


def get_rgb(tb_chips):
    data = tb_chips.get_data()
    rgb = data[ [g_red_idx,g_green_idx,g_blue_idx], g_time_idx, :, : ]
    rgb = np.transpose(rgb, (1,2,0))
    return rgb

# TODO move to utils
def normalize_feats(features):
    if len(features.shape) != 3:
        raise RuntimeError("Expecting features input to have 3 dimensions")
    for k in range(features.shape[2]):
        c = features[:,:,k]
        features[:,:,k] = (c - np.min(c)) / (np.max(c) - np.min(c))
    return features

def write_cats_and_features(region, rgb, features, cdl, cat_dict):
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

    plt.savefig( pj(HOME, "Training/cropnet/test_samples/features_%s.png" \
            % (region)) )


def main(args):
    bbox = [args.subregion_x, args.subregion_y,
            args.subregion_x + args.subregion_size,
            args.subregion_y + args.subregion_size]
    data_loader,tb_chips,cdl_file_path = get_data_loader(args.region, bbox)
    model = load_ae_model(args.model_path, args.model_name, chip_size=19,
            bneck_size=3, base_nchans=16) # TODO

    rgb = get_rgb(tb_chips)
    features = get_features(model, data_loader, bbox)
    features = normalize_feats(features)
    cdl = get_cdl_chip(cdl_file_path, bbox)
    cdl,cat_dict = transform_cdl(cdl)
    write_cats_and_features(args.region, rgb, features, cdl, cat_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default="ark",
            choices=["ark", "sd", "ohio", "vai"])
    parser.add_argument("--model-path", type=str, 
            default=pj(HOME, "Training/cropnet/sessions/session_07/models/" \
                    "model.pkl"))
    parser.add_argument("--model-name", type=str, default="CropNetFCAE")

    parser.add_argument("--subregion-x", type=int, default=0)
    parser.add_argument("--subregion-y", type=int, default=0)
    parser.add_argument("--subregion-size", type=int, default=20)

    args = parser.parse_args()
    main(args)

