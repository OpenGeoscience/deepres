"""
Some utilities for cropnet
"""

import gdal
import numpy as np
import os
import re
import shutil
import torch

from collections import OrderedDict

# ml_utils imports
from pyt_utils.encoder import compute_features

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


g_cat_pct_thresh = 0.05 # Pct in image to get recognized in legend
g_num_spectral = 19
g_time_start_idx = 7
g_time_end_idx = 26
g_hlstb_stub = "hls_tb_%s_%d_%d_%d_%d.npy"


# Input:
#   bbox: A bounding box in (x0,y0,x1,y1) format
# Output:
#   The area of the bounding box
def bbox_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

# Input: file name that has exactly one substring of the form _<n>_<n>_<n>_<n>,
# corresponding to a [x0, y0, x1, y1] bounding box.
def get_bbox_from_file_path(file_path):
    file_name = os.path.basename( os.path.abspath(file_path) )
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

def get_cat_dict(cdl):
    clut = make_clut()
    cat_dict = OrderedDict()
    for i in range(256):
        if np.sum(i==cdl) / cdl.size > g_cat_pct_thresh:
            cat_dict[i] = clut[i]
    return cat_dict

def get_cat_list(cats_opt):
    if cats_opt == "original4":
        cats = [3, 4, 5, 190]
    elif cats_opt == "top5":
        cats = [5, 1, 176, 0, 121]
    elif cats_opt == "top10":
        cats = [5, 1, 176, 0, 121, 141, 3, 24, 36, 61]
    elif cats_opt == "top25":
        cats = [5, 1, 176, 0, 121, 141, 3, 24, 36, 61, 122, 111, 123, 2, 69,
                190, 37, 142, 195, 54, 152, 76, 124, 4, 58]
    elif cats_opt == "top50":
        cats = [5, 1, 176, 0, 121, 141, 3, 24, 36, 61, 122, 111, 123, 2, 69,
                190, 37, 142, 195, 54, 152, 76, 124, 4, 58, 26, 28, 75, 6, 33,
                205, 13, 10, 131, 42, 77, 23, 27, 44, 50, 66, 21, 7, 8, 59, 
                9, 143, 48, 12, 11]
    elif cats_opt == "all":
        cats = list(range(256))
    else:
        raise RuntimeError("Unrecognized cats option, %s" % (cats_opt))
    return cats

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

def get_hls_subregions_all(region, hls_dir, bbox, saver=None):
    timepts_at_band = []
    for b in range(1,g_num_spectral+1):
        if saver is not None:
            saver_b = lambda regn,t : saver(regn, t, b)
        else:
            saver_b = None
        timepts_at_band.append( get_hls_subregions_by_band(region, b, hls_dir,
            bbox, saver_b) )
    return timepts_at_band

def get_hls_subregions_by_time(region, timepoint, hls_dir, bbox, saver=None):
    regions = []
    for b in range(1,g_num_spectral+1):
        path = pj(hls_dir, "hls_cls_%s_%02d.tif" % (region, b))
        img = gdal.Open(path)
        if img is None:
            raise RuntimeError("Image %s not found" % (path))
        layer = img.GetRasterBand(timepoint)
        region_tp = layer.ReadAsArray()
        region_tp = region_tp[ bbox[0]:bbox[2], bbox[1]:bbox[3] ]
        if saver is not None:
            regions.append(region_tp)
        else:
            saver(region_tp, path)
    return regions

def get_hls_subregions_by_band(region, band, hls_dir, bbox, saver=None):
    path = pj(hls_dir, "hls_cls_%s_%02d.tif" % (region, band))
    regions = []
    img = gdal.Open(path)
    if img is None:
        raise RuntimeError("Image %s not found" % (path))
    for t in range(g_time_start_idx, g_time_end_idx):
        layer = img.GetRasterBand(t)
        region_tp = layer.ReadAsArray()
        region_tp = region_tp[ bbox[0]:bbox[2], bbox[1]:bbox[3] ]
        if saver is None:
            regions.append(region_tp)
        else:
            saver(region_tp, t)
    return regions

def load_tb_chips(region, tbchips_dir, bbox_src, bbox=None):
    tb_chips = np.load( pj(tbchips_dir, g_hlstb_stub % (region, bbox_src[0],
        bbox_src[1], bbox_src[2], bbox_src[3])))
    if bbox is None:
        bbox = np.zeros(4, np.int)
        bbox[2] = bbox_src[2] - bbox_src[0]
        bbox[3] = bbox_src[3] - bbox_src[1]
    tb_chips = tb_chips[:, :, bbox[0]:bbox[2], bbox[1]:bbox[3]]
    return tb_chips

# Obviously, this is tied very closely to the specific format of this file
def make_clut(file_path=pj(HOME, 
    "Repos/OpenGeoscience/deepres/2016_cdl_color.txt")):
    color_dict = OrderedDict()
    reading_cat_vals = True
    with open(file_path) as fp:
        next(fp)
        for line in fp:
            line = line.strip()
            if len(line) == 0:
                reading_cat_vals = False
                continue
            if reading_cat_vals:
                cat = int( line[ : line.index(":") ] )
                rgb = line[ line.index(":")+2 : ]
                rgb = [int(x) for x in rgb.split(",")][:3]
                color_dict[cat] = OrderedDict()
                color_dict[cat]["rgb"] = rgb
            else:
                if "index=" in line:
                    beg = line.index("index=") + len("index=") + 1
                    end = line.index(">") - 1
                    cat = int( line[beg:end] )
                    next(fp)
                    line2=next(fp)
                    beg = line2.index(">") + 1
                    end = line2.index("<", beg)
                    name = line2[beg:end]
                    color_dict[cat]["name"] = name
    return color_dict

def normalize_feats(features):
    if len(features.shape) != 3:
        raise RuntimeError("Expecting features input to have 3 dimensions")
    for k in range(features.shape[2]):
        c = features[:,:,k]
        features[:,:,k] = (c - np.min(c)) / (np.max(c) - np.min(c))
    return features

def save_tb_chips(region, hls_dir, tb_chips, bbox_src, bbox):
    bbox = list(bbox)
    bbox[0] += bbox_src[0]
    bbox[1] += bbox_src[1]
    bbox[2] += bbox_src[0]
    bbox[3] += bbox_src[1]
    np.save( pj(hls_dir, g_hlstb_stub % (region, bbox[0], bbox[1], bbox[2],
        bbox[3])), tb_chips )

# Convert the cdl file into an rgb image with appropriate colors and labels
def transform_cdl(cdl):
    cat_dict = get_cat_dict(cdl)
    keys = list(cat_dict.keys())
    h,w = cdl.shape[:2]
    cdl_rgb = np.zeros((h,w,3))
    for i in range(256):
        if i in keys:
            cdl_rgb[ cdl==i, : ] = np.array( cat_dict[i]["rgb"] )
        else:
            cdl_rgb[ cdl==i, : ] = np.zeros(3)
    cdl_rgb /= 256.0
    return cdl_rgb,cat_dict


