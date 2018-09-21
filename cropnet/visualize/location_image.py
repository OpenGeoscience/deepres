"""
This function shows the 19x19 'images' corresponding to the different categories
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

from utils import get_chip_bbox

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


g_num_spectral = 19
g_time_start_idx = 7
g_time_end_idx = 26
g_hls_stub = "hls_cls_ark_time%d_band%d_%d_%d_%d_%d.npy" # TODO "ark"


def get_cdl_chip(cdl_file, bbox):
    cdl = np.load(cdl_file)
    if cdl.dtype != np.uint8:
        raise RuntimeError("Expecting type np.uint8 for %s, got %s" \
                % (cdl_file, cdl.dtype))
    return cdl[ bbox[0]:bbox[2], bbox[1]:bbox[3] ]

def get_hls_chips(hls_dir, bbox, bbox_src):
    hls_4d = []
    for b in range(1, g_num_spectral+1):
        band = []
        for i,t in enumerate( range(g_time_start_idx, g_time_end_idx) ):
            file_name = g_hls_stub % (t, b, bbox_src[0], bbox_src[1], 
                    bbox_src[2], bbox_src[3])
            hls = np.load( pj(hls_dir, file_name) )
            hls_chip = hls[ bbox[0]:bbox[2], bbox[1]:bbox[3] ]
            band.append(hls_chip)
        hls_4d.append( np.array(band) )
    hls_4d = np.array(hls_4d)

    tb_chips = []
    bbox_ht = bbox[2] - bbox[0]
    bbox_wd = bbox[3] - bbox[1]
    for i in range(bbox_ht):
        tb_chips_i = []
        for j in range(bbox_wd):
            tb_chip_ij = hls_4d[:, :, i, j]
            tb_chips_i.append( tuple( np.squeeze(tb_chip_ij) ) )
        tb_chips.append(tb_chips_i)
            
    return tb_chips
            
def make_figure(output_dir, cdl_chip, tb_chips):
    nrows,chip_cols = cdl_chip.shape
    ncols = 2 * chip_cols + 2
    grid_sz = (nrows, ncols)
    gridspec.GridSpec(nrows, ncols)

    plt.subplot2grid(grid_sz, (0,0), rowspan=nrows, colspan=nrows)
    plt.imshow(cdl_chip, cmap=plt.cm.gray)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale("off")

    for b,band in enumerate(tb_chips):
        for idx,chip in enumerate(band):
            i = b
            j = chip_cols + 2 + idx
            plt.subplot2grid(grid_sz, (i,j), rowspan=1, colspan=1)
            plt.imshow(chip, cmap=plt.cm.gray)
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.autoscale("off")
    plt.savefig( pj(output_dir, "cat_and_hls_comp.png") )


def main(args):
    bbox = get_chip_bbox(args.chip_x, args.chip_y, args.chip_size)
    bbox_src = get_chip_bbox(args.src_image_x, args.src_image_y, 
            args.src_image_size)
    cdl_chip = get_cdl_chip(args.ground_truth_file, bbox)
    tb_chips = get_hls_chips(args.hls_dir, bbox, bbox_src)
    make_figure(args.output_dir, cdl_chip, tb_chips)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cdl", "--ground-truth", dest="ground_truth_file",
            type=str,
            default=pj(HOME, "Datasets/HLS/test_imgs/cdl/" \
                    "cdl_2016_neAR_0_0_500_500.npy"))
    parser.add_argument("--hls-dir", type=str,
            default=pj(HOME, "Datasets/HLS/test_imgs/hls"))
    parser.add_argument("-o", "--output-dir", type=str,
            default=pj(HOME, "Training/cropnet/test_samples"))
    parser.add_argument("-x", "--chip-x", type=int, default=0,
            help="Chip top coordinate")
    parser.add_argument("-y", "--chip-y", type=int, default=0,
            help="Chip left coordinate")
    parser.add_argument("--chip-size", type=int, default=20)
    parser.add_argument("--src-image-x", type=int, default=0,
            help="Source chip top coordinate")
    parser.add_argument("--src-image-y", type=int, default=0,
            help="Source chip left coordinate")
    parser.add_argument("--src-image-size", type=int, default=500)
    args = parser.parse_args()
    main(args)

