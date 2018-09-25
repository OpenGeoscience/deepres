"""
Datasets for cropnet
"""

import csv
import logging
import numpy as np
import os
import torch
import torchvision as tv

from collections import OrderedDict
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from visualize.location_image import get_chip_bbox, get_hls_chips


logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("Image").setLevel(logging.WARNING)
logging.getLogger("PIL.Image").setLevel(logging.WARNING)

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")

#bbox = get_chip_bbox(args.chip_x, args.chip_y, args.chip_size)
#bbox_src = get_chip_bbox(args.src_image_x, args.src_image_y,
#        args.src_image_size)
#tb_chips = get_hls_chips(args.hls_dir, bbox, bbox_src)

# Time-Band Chips, where Band is spectral band
class TBChips(Dataset):
    def __init__(self, data_dir, src_image_x=0, src_image_y=0, 
            src_image_size=500):
        self._data_dir = data_dir
        self._N = None
        self._src_image_x = src_image_x
        self._src_image_y = src_image_y
        self._src_image_size = src_image_size
        self._tb_chips = None # We're going to hold these in memory for now

        self._get_tb_chips()

    def __getitem__(self, index):
        i = index // self._src_image_size
        j = index % self._src_image_size
        tb_chip = np.squeeze( self._tb_chips[i,j] )
#        tb_chip = tb_chip / 255.0
#        tb_chip = tv.transforms.ToTensor()(tb_chip)
        tb_chip = torch.FloatTensor(tb_chip).unsqueeze(0)
        return tb_chip,tb_chip

    def __len__(self):
        return self._N

    def _get_tb_chips(self):
        bbox = get_chip_bbox(self._src_image_x, self._src_image_y, 
                self._src_image_size)
        self._tb_chips = get_hls_chips(self._data_dir, bbox)
        self._N = len(self._tb_chips)
