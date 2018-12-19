"""
Create smaller versions of the hls/ground truth files for testing/visualization
"""

import argparse
import cv2
import gdal
import numpy as np
import os
import platform
import shutil

from make_images import make_and_save_cdl, make_and_save_cdl,\
        make_and_save_tbchips
from utils import get_chip_bbox

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")
if platform.node() == "matt-XPS-8900":
    DATA = HOME
else:
    DATA = "/media/data"


g_num_spectral = 19
g_time_start_idx = 7
g_time_end_idx = 26
g_hls_stub = "hls_cls_ark_time%d_band%d_%d_%d_%d_%d.npy" # TODO "ark"

g_chip_size = 1000  # 1000x1000 pix
g_region_subdirs = ["ark", "sd", "ohio", "vai"]

def main(args):
    for subdir in g_region_subdirs:
        input_dir = pj(args.input_supdir, subdir)
        output_dir = pj(args.output_supdir, subdir)
        if pe(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        gt_file = pj(args.ground_truth_dir, cdl_file)
            # TODO select cdl file based on region
        for i in range(3): # TODO
            xstart = i*g_chip_size
            for j in range(3): # TODO
                ystart = j*g_chip_size
                bbox = get_chip_bbox(xstart, ystart, g_chip_size)
                make_and_save_hls(output_dir, args.hls_dir, bbox)
                make_and_save_cdl(output_dir, args.ground_truth_file, bbox)
                    make_and_save_tbchips(output_dir, bbox)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-supdir", type=str, 
            default=pj(DATA, "Datasets/HLS"))
    parser.add_argument("-o", "--output-supdir", type=str,
            default=pj(DATA, "Datasets/HLS/preprocess"))
    args = parser.parse_args()
    main(args)

