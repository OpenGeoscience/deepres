"""
This function displays B&W band-specific imagery en regard with the CDL class-
ification.
"""

import argparse
import gdal
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import shutil

from utils import get_cdl_subregion, get_chip_bbox, get_hls_subregions_by_band,\
        get_cdl_subregions_by_time

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


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

