"""
Datasets for cropnet
"""

import csv
import cv2
import logging
import numpy as np
import os
import torch
import torchvision as tv

from collections import OrderedDict
from PIL import Image
from skimage.transform import resize

# pytorch imports
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# ml_utils imports

# local imports
from utils import get_chip_bbox, load_tb_chips


logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("Image").setLevel(logging.WARNING)
logging.getLogger("PIL.Image").setLevel(logging.WARNING)

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


# 224x224 patches to feed into pre-trained ResNet, classify center pixel
class RGBPatches(Dataset):
    def __init__(self, data_dir_or_file, labels_dir_or_file, upsample=2, 
            ingest_patch_size=224):
        self._data_dir = None
        self._data_img = None
        self._img_ht = None
        self._img_wd = None
        self._ingest_patch_size = ingest_patch_size
        self._labels_dir = None
        self._labels_img = None
        self._pad = None
        self._upsample = upsample

        self._pad = self._ingest_patch_size // self._upsample

        if os.path.isdir(data_dir_or_file):
            self._data_dir = data_dir_or_file
            raise NotImplementedError()
        else:
            self._load_rgb(data_dir_or_file)

        if os.path.isdir(labels_dir_or_file):
            self._data_dir = labels_dir_or_file
            raise NotImplementedError()
        else:
            self._load_labels(labels_dir_or_file)

        self._check_dims()

    def __getitem__(self, index):
        i = self._pad + index % self._img_ht  # Column major
        j = self._pad + index // self._img_ht
        i_beg = i - self._pad
        i_end = i + self._pad
        j_beg = j - self._pad
        j_end = j + self._pad
        raw_patch = self._data_img[i_beg:i_end, j_beg:j_end]
        resize_dims = (self._ingest_patch_size, self._ingest_patch_size)
        patch = cv2.resize(raw_patch,resize_dims,interpolation=cv2.INTER_CUBIC)
        label = self._labels_img[i_beg][j_beg]
        return patch,label

    def __len__(self):
        return self._img_ht * self._img_wd

    def _check_dims(self):
        label_h,label_w = self._labels_img.shape[:2]
        h,w = self._data_img.shape[:2]
        if label_h+2*self._pad != h or label_w+2*self._pad != w:
            raise RuntimeError("Inconsistent data, label matrices; (%d, %d) " \
                    "with total padding=%d vs. (%d, %d)." \
                    % (h, w, 2*self._pad, label_h, label_w))

    def _load_labels(self, data_npy_file):
        self._labels_img = np.load(data_npy_file)
        if self._labels_img is None:
            raise RuntimeError("Image file %s does not exist or is not a valid"\
                    " image")

    def _load_rgb(self, data_npy_file):
        self._data_img = np.load(data_npy_file)
        if self._data_img is None:
            raise RuntimeError("Image file %s does not exist or is not a valid"\
                    " image")
        self._img_ht,self._img_wd = self._data_img.shape[:2]
        pad_shape = ((self._pad, self._pad), (self._pad, self._pad), (0,0))
        self._data_img = np.pad(self._data_img/256, pad_shape, mode="reflect")


# Time-Band Chips, where Band is spectral band
class TBChips(Dataset):
    def __init__(self, data_dir_or_file, src_image_x=0, src_image_y=0, 
            src_image_size=500):
        self._data_dir = None
        self._data_npy_file = None
        self._N = None
        self._src_image_x = src_image_x
        self._src_image_y = src_image_y
        self._src_image_size = src_image_size
        self._tb_chips = None # We're going to hold these in memory for now

        if os.path.isdir(data_dir_or_file):
            self._data_dir = data_dir_or_file
            self._get_tb_chips()
        else:
            self._load_tb_chips_from_npy(data_dir_or_file)

    def __getitem__(self, index):
        i = index // self._src_image_size
        j = index % self._src_image_size
        tb_chip = np.squeeze( self._tb_chips[i,j] )
        tb_chip = torch.FloatTensor(tb_chip).unsqueeze(0)
        return tb_chip,tb_chip

    def __len__(self):
        return self._N

    def get_data(self):
        return self._tb_chips

    def get_image_bbox(self):
        x0,y0 = self._src_image_x,self._src_image_y
        x1,y1 = x0+self._src_image_size,y0+self._src_image_size
        return (x0,y0,x1,y1)

    def _get_tb_chips(self):
        src_bbox = get_chip_bbox(self._src_image_x, self._src_image_y, 
                self._src_image_size)
        self._tb_chips = load_tb_chips(self._data_dir, src_bbox)
        # TODO load_tb_chips does virtually exactly what the version here does, 
        # just loads teh npy file directly.  Doesn't create it
        self._N = self._tb_chips.shape[0] * self._tb_chips.shape[1]

    def _load_tb_chips_from_npy(self, data_file):
        full_tb_chips = np.load(data_file)
        x0,y0,x1,y1 = self.get_image_bbox()
        self._tb_chips = full_tb_chips[y0:y1, x0:x1]
        self._N = self._tb_chips.shape[0] * self._tb_chips.shape[1]

