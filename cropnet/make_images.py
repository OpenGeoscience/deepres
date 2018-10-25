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
        get_hls_subregions_by_band, get_hls_subregions_by_time, save_tb_chips

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


g_num_spectral = 19
g_time_start_idx = 7
g_time_end_idx = 26
g_hls_stub = "hls_cls_ark_time%d_band%d_%d_%d_%d_%d.npy" # TODO "ark"

def _hls_saver(region, path_stub, t, b, bbox):
    print("_hls_saver 1")
    path_stub = path_stub % (t, b, bbox[0], bbox[1], bbox[2], bbox[3])
    print("_hls_saver 2")
    img_path_png = path_stub + ".png"
    print("_hls_saver 3")
    img_path_npy = path_stub + ".npy"
    print("_hls_saver 4")
    cv2.imwrite(img_path_png, region)
    print("_hls_saver 5")
    np.save(img_path_npy, region)
    print("Saved %s and %s to disk" % (img_path_png, img_path_npy))
    
def _map_to_uniform(band):
    band_shape = band.shape
    b = band.flatten()
    N = len(b)
    sb_pairs = sorted(enumerate(b), key=lambda x: x[1])
    idxs,_ = zip(*sb_pairs)
    u_pairs = zip(idxs, np.linspace(0, 1, N))
    su_pairs = sorted(u_pairs, key=lambda x: x[0])
    _,su = zip(*su_pairs)
    su = np.array(list(su))
    u_band = su.reshape(*band_shape)
    return u_band


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
    
def make_and_save_tbchips(hls_dir, bbox_src, bbox=None):
    if bbox is None:
        bbox = [0, 0, bbox_src[2]-bbox_src[0], bbox_src[3]-bbox_src[1]]
    hls_4d = []
    print("Loading original hls images...")
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
    print("...Done")
    print(hls_4d.shape)

    # For a 1000x1000 source image, hls_4d.shape is [19, 19, 1000, 1000],
    # [ band, time, ht, wd ]
    print("Normalizing band data...")
    for b in range(g_num_spectral):
        band = hls_4d[b,:,:,:]
        band[:,:,:] = _map_to_uniform(band)
    print("...Done")

    save_tb_chips(hls_dir, hls_4d, bbox_src, bbox)
    return hls_4d

def main(args):
    bbox = get_chip_bbox(args.image_x, args.image_y, args.image_size)
    if not args.tb_chips_only:
        make_and_save_hls(args.output_supdir, args.hls_dir, bbox)
        make_and_save_cdl(args.output_supdir, args.ground_truth_file, bbox)
    make_and_save_tbchips(pj(args.output_supdir, "hls"), bbox)
    print("Done, files written to %s" % (args.output_supdir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cdl", "--ground-truth", dest="ground_truth_file",
            type=str,
            default=pj(HOME, "Datasets/HLS/ground_truth/cdl_2016_neAR.tif"))
    parser.add_argument("--hls-dir", type=str,
            default=pj(HOME, "Datasets/HLS/ark"))
    parser.add_argument("-o", "--output-supdir", type=str,
            default=pj(HOME, "Datasets/HLS/test_imgs"))

    parser.add_argument("--tb-chips-only", action="store_true",
            help="Set this if the requisite hls files have already been " \
                    "generated and you just want to generate the 4D file.")
    parser.add_argument("-x", "--image-x", type=int, default=0,
            help="Image top coordinate")
    parser.add_argument("-y", "--image-y", type=int, default=0,
            help="Image left coordinate")
    parser.add_argument("--image-size", type=int, default=500)
    args = parser.parse_args()
    main(args)

