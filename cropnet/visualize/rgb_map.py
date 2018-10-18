"""
This uses a pre-existing model to create a false color map using the results 
from the autoencoder.  This false color image is then to be fed into a 
traditional image segmentation routine
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import shutil

from location_image import get_hls_chips
from utils import get_chip_bbox

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


def main(args):
    encodings = get_encodings(args.data_dir, args.model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-dir", type=str,
            default=pj(HOME, "Datasets/HLS/test_imgs/hls"))
    parser.add_argument("--model-path", type=str, 
            default=pj(HOME, "Training/cropnet/sessions/session_07/models/" \
                    "model.pkl"))
    args = parser.parse_args()
    main(args)

