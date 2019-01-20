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
from utils import get_features, get_bbox_from_file_path, normalize_feats

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


g_regions = ["ark", "ohio", "sd", "vai"]


def _extract_session_dir(model_path):
    return os.path.dirname( os.path.dirname(model_path) )

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


def make_full_feat_map(region, model, cfg):
    session_dir = _extract_session_dir(cfg["model_path"])
    feat_maps_dir = pj(session_dir, "feat_maps")
    if not pe(feat_maps_dir):
        os.makedirs(feat_maps_dir)
    feat_map = np.zeros((3000,3000,3)) # TODO
    d = cfg["data_dir"]
    for p in [pj(d,f) for f in os.listdir(d) if f.startswith("hls_tb_"+region)]:
        tb_chips = TBChips(data_dir=d, tiles_per_cohort=1,
                data_file=os.path.basename(p))
        data_loader = DataLoader(dataset=tb_chips,
                shuffle=False,
                batch_size=cfg["batch_size"],
                num_workers=cfg["num_workers"])
        bbox = get_bbox_from_file_path(p)
        sz = bbox[2] - bbox[0]
        feats = get_features(model, data_loader, [0, 0, sz, sz])
        feat_map[ bbox[0]:bbox[2], bbox[1]:bbox[3], : ] = feats
    feat_map = normalize_feats(feat_map)
    save_path = pj(feat_maps_dir, "%s_feat_map.npy" % (region))
    np.save(save_path, feat_map)
    cv2.imwrite(save_path[:-4] + ".png", feat_map*255)

def main(args):
    cfg = vars(args)
    model = load_ae_model(args.model_path, args.model_name, chip_size=19,
            bneck_size=3, base_nchans=16) # TODO
    for region in g_regions:
        make_full_feat_map(region, model, cfg)

#        
#    bbox = [args.subregion_x, args.subregion_y,
#            args.subregion_x + args.subregion_size,
#            args.subregion_y + args.subregion_size]
#    data_loader,tb_chips,cdl_file_path = get_data_loader(args.region, bbox)
#
#    rgb = get_rgb(tb_chips)
#    features = get_features(model, data_loader, bbox)
#    features = normalize_feats(features)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-name", type=str, default="CropNetCAE",
            choices=["CropNetCAE", "CropNetFCAE"])
    parser.add_argument("--model-path", type=str, 
            default=pj(HOME, "Training/cropnet/sessions/session_02/models/" \
                    "CropNetCAE.pkl"))
    parser.add_argument("-d", "--data-dir", type=str,
            default=pj(HOME, "Datasets/HLS/tb_data/all/hls"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    main(args)

