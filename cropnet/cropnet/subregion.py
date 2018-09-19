"""
This function displays B&W band-specific imagery en regard with the CDL class-
ification.
"""

import argparse
import cv2
import gdal
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import pandas as pd
import shutil

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")

"""
For CDL
West_Bounding_Coordinate: -127.8873
East_Bounding_Coordinate: -74.1585
North_Bounding_Coordinate: 47.9580
South_Bounding_Coordinate: 23.1496

CDL
Corner Coordinates:
Upper Left  (-2356095.000, 3172605.000) (127d53'13.96"W, 47d57'30.26"N)
Lower Left  (-2356095.000,  276915.000) (118d45'10.81"W, 22d56'24.97"N)
Upper Right ( 2258235.000, 3172605.000) ( 65d20'43.83"W, 48d14'43.04"N)
Lower Right ( 2258235.000,  276915.000) ( 74d 9'30.37"W, 23d 8'58.53"N)
Center      (  -48930.000, 1724760.000) ( 96d34' 0.39"W, 38d33' 4.57"N)

HLS
Corner Coordinates:
Upper Left  (  699960.000, 4000020.000) ( 90d46'40.97"W, 36d 7'27.42"N)
Lower Left  (  699960.000, 3890220.000) ( 90d48'18.97"W, 35d 8' 6.04"N)
Upper Right (  809760.000, 4000020.000) ( 89d33'34.49"W, 36d 5'43.68"N)
Lower Right (  809760.000, 3890220.000) ( 89d36' 6.02"W, 35d 6'25.99"N)
Center      (  754860.000, 3945120.000) ( 90d11'10.11"W, 35d37' 1.34"N)

"""

g_num_spectral = 19

def get_cdl_subregion(img_path, bbox):
    img = gdal.Open(img_path)
    layer = img.GetRasterBand(1)
    region = layer.ReadAsArray()
    return region[ bbox[0]:bbox[2], bbox[1]:bbox[3] ]

def get_chip_bbox(chip_x, chip_y, chip_size):
    return (chip_x, chip_y, chip_x+chip_size, chip_y+chip_size)

def get_hls_subregions_by_time(timepoint, hls_dir, bbox):
    regions = []
    for i in range(1,20):
        path = pj(hls_dir, "hls_cls_ark_%02d.tif" % (i))
        img = gdal.Open(path)
        layer = img.GetRasterBand(timepoint)
        region = layer.ReadAsArray()
        regions.append( region[ bbox[0]:bbox[2], bbox[1]:bbox[3] ] )
    return regions

def get_hls_subregions_by_band(band, hls_dir, bbox):
    path = pj(HOME, "Datasets/HLS/ark/hls_cls_ark_%02d.tif" % (band))
    regions = []
    img = gdal.Open(path)
    for i in range(7,26):
        layer = img.GetRasterBand(i)
        region = layer.ReadAsArray()
        regions.append( region[ bbox[0]:bbox[2], bbox[1]:bbox[3] ] )
    return regions

def make_cat_hist(region):
    cats = np.unique(region)

def make_figure(hls_by_band, hls_by_time, cdl_subregion, band, timepoint):
    nrows = 9
    ncols = 5
    grid_sz = (nrows, ncols)

    def _make_panel(hls_subregions, panel_rows, row_offset):
        plt.subplot2grid(grid_sz, (row_offset,0), rowspan=1, colspan=1)
        plt.imshow(cdl_subregion, cmap=plt.cm.gray)

        for ct,hls_subregion in enumerate(hls_subregions):
            i = ((ct+1) % panel_rows) + row_offset
            j = (ct+1) // panel_rows
            plt.subplot2grid(grid_sz, (i,j), rowspan=1, colspan=1)
            plt.imshow(hls_subregion, cmap=plt.cm.gray)
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
           # ax.autoscale("off")

    gridspec.GridSpec(nrows, ncols)
    _make_panel(hls_by_band, 4, 0)
    _make_panel(hls_by_time, 4, 5)
#    plt.title("Top spectral, bottom timepoint")
    plt.savefig( pj(HOME, "Training/cropnet/test_samples/" \
            "cat_and_hls_band%02d_time%02d.png" % (band, timepoint)) )
    

def main(args):
    bbox = get_chip_bbox(args.chip_x, args.chip_y, args.chip_size)
    hls_by_band = get_hls_subregions_by_band(args.band, args.hls_dir, bbox)
    hls_by_time = get_hls_subregions_by_time(args.timepoint, args.hls_dir, bbox)
    cdl_subregion = get_cdl_subregion(args.ground_truth_file, bbox)
    make_cat_hist(cdl_subregion)
    make_figure(hls_by_band, hls_by_time, cdl_subregion, args.band, 
        args.timepoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cdl", "--ground-truth", dest="ground_truth_file",
            type=str, 
            default=pj(HOME, "Datasets/HLS/ground_truth/cdl_2016_neAR.tif"))
    parser.add_argument("--hls-dir", type=str,
            default=pj(HOME, "Datasets/HLS/ark"))
    parser.add_argument("-b", "--band", type=int, default=5,
            help="Which spectral band to show") 
    parser.add_argument("-t", "--timepoint", type=int, default=14,
            help="Which time index to show, 1-25 incl.")
    parser.add_argument("-x", "--chip-x", type=int, default=0,
            help="Chip top coordinate")
    parser.add_argument("-y", "--chip-y", type=int, default=0,
            help="Chip left coordinate")
    parser.add_argument("--chip-size", type=int, default=100)
    args = parser.parse_args()
    main(args)

