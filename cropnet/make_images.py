"""
Create smaller versions of the hls/ground truth files for testing/visualization
"""

import argparse
import cv2
import gdal
import numpy as np
import os
import shutil

from utils import get_cdl_subregion, get_chip_bbox, get_hls_subregions_all, \
        get_hls_subregions_by_band, get_hls_subregions_by_time

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


def _hls_saver(region, path_stub, t, b, bbox):
    path_stub = path_stub % (t, b, bbox[0], bbox[1], bbox[2], bbox[3])
    img_path_png = path_stub + ".png"
    img_path_npy = path_stub + ".npy"
    cv2.imwrite(img_path_png, region)
    np.save(img_path_npy, region)
    print("Saved %s and %s to disk" % (img_path_png, img_path_npy))
    

def make_and_save_cdl(output_supdir, cdl_file, bbox):
    img = get_cdl_subregion(cdl_file, bbox)
    cdl_name_stub = os.path.splitext( os.path.basename(cdl_file) )[0]
    img_name_stub = cdl_name_stub + "_%d_%d_%d_%d" % (bbox[0], bbox[1], bbox[2],
            bbox[3])
    img_name_png = img_name_stub + ".png"
    img_name_npy = img_name_stub + ".npy"
    cdl_dir = pj(output_supdir, "cdl")
    if not pe(cdl_dir):
        os.makedirs(cdl_dir)
    cv2.imwrite(pj(cdl_dir, img_name_png), img)
    np.save(pj(cdl_dir, img_name_npy), img)

def make_and_save_hls(output_supdir, hls_dir, bbox):
    hls_out_dir = pj(output_supdir, "hls")
    if not pe(hls_out_dir):
        os.makedirs(hls_out_dir)
    path_stub = pj(hls_out_dir, "hls_cls_ark_time%d_band%d_%d_%d_%d_%d")
    saver = lambda region,t,b : _hls_saver(region, path_stub, t, b, bbox)
    get_hls_subregions_all(hls_dir, bbox, saver)
    

def main(args):
    bbox = get_chip_bbox(args.image_x, args.image_y, args.image_size)
    make_and_save_hls(args.output_supdir, args.hls_dir, bbox)
    make_and_save_cdl(args.output_supdir, args.ground_truth_file, bbox)
    print("Done, files written to %s" % (args.output_supdir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-supdir", type=str,
            default=pj(HOME, "Datasets/HLS/test_imgs"))
    parser.add_argument("--cdl", "--ground-truth", dest="ground_truth_file",
            type=str,
            default=pj(HOME, "Datasets/HLS/ground_truth/cdl_2016_neAR.tif"))
    parser.add_argument("--hls-dir", type=str,
            default=pj(HOME, "Datasets/HLS/ark"))
    parser.add_argument("-x", "--image-x", type=int, default=0,
            help="Image top coordinate")
    parser.add_argument("-y", "--image-y", type=int, default=0,
            help="Image left coordinate")
    parser.add_argument("--image-size", type=int, default=500)
    args = parser.parse_args()
    main(args)

