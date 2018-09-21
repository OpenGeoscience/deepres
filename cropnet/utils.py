"""
Some utilities for the visualization scripts
"""

import gdal
import numpy as np
import os
import shutil

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


g_num_spectral = 19
g_time_start_idx = 7
g_time_end_idx = 26


def get_cdl_subregion(img_path, bbox):
    img = gdal.Open(img_path)
    layer = img.GetRasterBand(1)
    region = layer.ReadAsArray()
    return region[ bbox[0]:bbox[2], bbox[1]:bbox[3] ]

def get_chip_bbox(chip_x, chip_y, chip_size):
    return (chip_x, chip_y, chip_x+chip_size, chip_y+chip_size)

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

