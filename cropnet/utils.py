"""
Some utilities for the visualization scripts
"""

import gdal
import numpy as np
import os
import re
import shutil
import torch

from cropnet.models import CropNetFCAE

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


g_num_spectral = 19
g_time_start_idx = 7
g_time_end_idx = 26


# Input: file name that has exactly one substring of the form _<n>_<n>_<n>_<n>,
# corresponding to a [x0, y0, x1, y1] bounding box.
def get_bbox_from_file_name(file_name):
    file_name = os.path.basename(file_name)
    match = re.search(r"_\d+_\d+_\d+_\d+", file_name)
    if match is None:
        raise RuntimeError("Incorrect format of file name %s, must contain " \
                "substring of form _<n>_<n>_<n>_<n>" % (file_name))
    span = match.span(0)
    rematch = re.search(r"_\d+_\d+_\d+_\d+", file_name[ span[0]+1 : ])
    if rematch is not None:
        raise RuntimeError("Incorrect format of file name %s, substring of " \
                "form _<n>_<n>_<n>_<n> must be unique" % (file_name))
    bbox_str = file_name[ span[0] : span[1] ][1:]
    uscore = bbox_str.find("_")
    x0 = int( bbox_str[0 : uscore] )
    bbox_str = bbox_str[uscore+1:]
    uscore = bbox_str.find("_")
    y0 = int( bbox_str[0 : uscore] )
    bbox_str = bbox_str[uscore+1:]
    uscore = bbox_str.find("_")
    x1 = int( bbox_str[0 : uscore] )
    bbox_str = bbox_str[uscore+1:]
    y1 = int(bbox_str)
    return x0,y0,x1,y1

def get_cdl_subregion(img_path, bbox):
    img = gdal.Open(img_path)
    layer = img.GetRasterBand(1)
    region = layer.ReadAsArray()
    return region[ bbox[0]:bbox[2], bbox[1]:bbox[3] ]

def get_chip_bbox(chip_x, chip_y, chip_size):
    return (chip_x, chip_y, chip_x+chip_size, chip_y+chip_size)

def get_features(model, data_loader, bbox=None):
    features,_ = compute_features(model, data_loader, make_chip_list=False)
    if bbox is None:
        bbox = data_loader.dataset.get_image_bbox()
    size_x,size_y = bbox[2]-bbox[0],bbox[3]-bbox[1]
    num_chans = features.shape[1]
    features = features.reshape((size_y,size_x,num_chans))
    return features

def get_hls_subregions_all(hls_dir, bbox, saver=None):
    timepts_at_band = []
    for b in range(1,g_num_spectral+1):
        if saver is not None:
            saver_b = lambda region,t : saver(region, t, b)
        else:
            saver_b = None
        timepts_at_band.append( get_hls_subregions_by_band(b, hls_dir, bbox,
            saver_b) )
    return timepts_at_band

def get_hls_subregions_by_time(timepoint, hls_dir, bbox, saver=None):
    regions = []
    for b in range(1,g_num_spectral+1):
        path = pj(hls_dir, "hls_cls_ark_%02d.tif" % (b))
        img = gdal.Open(path)
        if img is None:
            raise RuntimeError("Tmage %s not found" % (path))
        layer = img.GetRasterBand(timepoint)
        region = layer.ReadAsArray()
        region = region[ bbox[0]:bbox[2], bbox[1]:bbox[3] ]
        if saver is not None:
            regions.append(region)
        else:
            saver(region, path)
    return regions

def get_hls_subregions_by_band(band, hls_dir, bbox, saver=None):
    path = pj(hls_dir, "hls_cls_ark_%02d.tif" % (band))
    regions = []
    img = gdal.Open(path)
    if img is None:
        raise RuntimeError("Tmage %s not found" % (path))
    for t in range(g_time_start_idx, g_time_end_idx):
        layer = img.GetRasterBand(t)
        region = layer.ReadAsArray()
        region = region[ bbox[0]:bbox[2], bbox[1]:bbox[3] ]
        if saver is None:
            regions.append(region)
        else:
            saver(region, t)
    return regions

def load_model(model_path, model_name, is_train=False, **kwargs):
    if model_name=="CropNetFCAE":
        model = CropNetFCAE(**kwargs)
    else:
        raise RuntimeError("Model %s not recognized" % (model_name))
    model.load_state_dict( torch.load(model_path) )
    if is_train:
        model = model.cuda().train()
    else:
        model = model.cuda().eval()
    return model

