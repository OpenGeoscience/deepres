"""
Datasets for cropnet
"""

import abc
import argparse
import csv
import cv2
import logging
import numpy as np
import os
import platform
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
from utils import (get_chip_bbox, load_tb_chips, make_clut, transform_cdl,
        get_bbox_from_file_path, bbox_area)


logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("Image").setLevel(logging.WARNING)
logging.getLogger("PIL.Image").setLevel(logging.WARNING)

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")
if platform.node() == "matt-XPS-8900":
    DATA = HOME
else:
    DATA = "/media/data"


# Base class
class CropNetBase(Dataset):
    def __init__(self, data_dir=None, labels_dir=None, tiles_per_cohort=10,
            **kwargs):
        super().__init__(**kwargs)
        self._cohort_ct = None
        self._data_dir = data_dir
        self._data_chunks = None
        self._data_paths = None
        self._item_ct = None
        self._labels_dir = labels_dir
        self._labels_imgs = None
        self._labels_paths = None
        self._N = None
        self._tile_cohorts = None
        self._tiles_per_cohort = tiles_per_cohort

        if self._data_dir is None:
            raise RuntimeError("Data directory not specified")

        self._get_paths()
        self._init_cohorts()
        self._load_new_cohort()
        self._calc_N()

    def __len__(self):
        return self._N

    def check_for_cohort_update(self):
        if self._item_ct >= self.items_per_cohort():
            self._load_new_cohort()

    def items_per_cohort(self):
        return self._N // len(self._tile_cohorts)

    def num_chunks(self):
        return len(self._data_paths)

    def _get_chunk_and_label(self, index):
        idx = index % self._tiles_per_cohort
        labels_img = None if self._labels_dir is None \
                else self._labels_imgs[idx]
        return self._data_chunks[idx], labels_img

    @abc.abstractmethod
    def _calc_N(self):
        pass

    def _get_paths(self):
        dd = self._data_dir
        self._data_paths = [pj(dd,f) for f in os.listdir(dd)]
        if self._labels_dir is not None:
            ld = self._labels_dir
            self._labels_paths = [pj(ld,f) for f in os.listdir(ld)]
            if len(self._data_paths) != len(self._labels_paths):
                raise RuntimeError("Different number of files in %s and %s" \
                        % (self._data_dir, self._labels_dir))
        # TODO Ensure that the labels files correspond exactly to the data 
        # files, don't rely on consistent alphabetization

    def _init_cohorts(self):
        N = len(self._data_paths)
        idxs = list(range(N))
        np.random.shuffle(idxs)
        self._tile_cohorts = []
        for i in range(N//self._tiles_per_cohort):
            self._tile_cohorts.append( idxs[:self._tiles_per_cohort] )
            idxs = idxs[self._tiles_per_cohort:]
        if len(idxs)>0:
            rem = self._tiles_per_cohort - len(idxs)
            rem_idxs = list(range(N))
            np.random.shuffle(rem_idxs)
            idxs += rem_idxs[:rem]
            self._tile_cohorts.append(idxs)
        self._cohort_ct = 0

    def _load_new_cohort(self):
        print("Loading next cohort of %d images and labels ..." \
                % (self._tiles_per_cohort))
        idxs = self._tile_cohorts[ self._cohort_ct ]
        self._data_chunks = []
        self._labels_imgs = []
        for idx in idxs:
            self._data_chunks.append( np.load(self._data_paths[idx]) )
            if self._labels_dir is not None:
                self._labels_imgs.append( np.load(self._labels_paths[idx]) )
        self._cohort_ct += 1
        self._item_ct = 0
        print("...Done")
        if self._cohort_ct == len(self._tile_cohorts):
            self._init_cohorts()

    def _update_item_ct(self):
        self._item_ct += 1

# 224x224 patches to feed into pre-trained ResNet, classifying whole patch 
# For this dataset the rgb images should be stitched together, i.e. combine
# all rgb feature outputs from a given area subject to memory constraints
class RGBPatches(CropNetBase):
    def __init__(self, cats={}, ingest_patch_size=224, mode="test", 
            **kwargs):
        super().__init__(tiles_per_cohort=100, **kwargs)
        self._cats = [3, 4, 5, 190] # TODO--just estimated from a single chip
        self._ingest_patch_size = ingest_patch_size
        self._label_mode = None
        self._mode = mode

    def __getitem__(self, index):
        rgb,label = self._get_chunk_and_label(index)
        ht,wd = rgb.shape[:2]

        if self._mode=="train": # TODO pre-calculate
            i_beg,j_beg,i_end,j_end = self._get_rand_bbox(ht,wd)
        else:
            # TODO accommodate a different # of chips per chunk
            # Right now assuming nice square chunks (rgb images)
            index = index % (ht * wd)
            sz = self._ingest_patch_size
            i = index % (ht // sz)
            j = index // (ht // sz)
            i_beg = i * sz
            j_beg = j * sz
            i_end = i_beg + sz
            j_end = j_beg + sz

        patch = self._data_img[i_beg:i_end, j_beg:j_end]
        patch = np.copy(patch)
        for k in range(patch.shape[2]): # TODO take care of this in stitching
            c = patch[:,:,k]
            patch[:,:,k] = (c - np.min(c)) / (np.max(c) - np.min(c))
        raw_label = self._labels_img[i_beg:i_end, j_beg:j_end]
        label = np.copy(raw_label)

        bgnd_mask = np.ones_like(label) > 0
        ct = 1
        for k in self._cats:
            mask_k = raw_label==k
            label[mask_k] = ct
            ct += 1
            bgnd_mask[mask_k] = False
        label[bgnd_mask] = 0

        label = torch.LongTensor([label]).squeeze()
        patch = Image.fromarray( np.uint8(patch*255) )
        transform = tv.transforms.ToTensor()
        return transform(patch),label 

    def get_num_cats(self):
        return len(self._cats) + 1

    # Currently this throws out the outer border, whatever exceeds n*sz pix
    def _calc_N(self):
        acc = 0
        sz = self._ingest_patch_size
        self._chips_per_chunk = []
        for p in self._data_paths:
            wd,ht = Image.open(p).shape
            n_wd = wd // sz
            n_ht = ht // sz
            self._chips_per_chunk.append(n_wd*n_ht)
            acc += n_wd * n_ht
        self._N = acc

    def _get_rand_bbox(self, ht, wd):
        sz = self._ingest_patch_size
        x,y = torch.rand(2).data
        i_beg = int( x*(wd-sz) )
        j_beg = int( y*(ht-sz) )
        return i_beg,j_beg,i_beg+sz,j_beg+sz
#
#
#    def _load_labels(self, data_npy_file):
#        print("Loading labels file %s ..." % (data_npy_file))
#        self._labels_img = np.load(data_npy_file)
#        if self._labels_img is None:
#            raise RuntimeError("Image file %s does not exist or is not a valid"\
#                    " image")
#        print("...Done.  Shape is %s" % repr(self._labels_img.shape))
#
#    def _load_rgb(self, data_npy_file):
#        print("Loading rgb file %s ..." % (data_npy_file))
#        self._data_img = np.load(data_npy_file)
#        if self._data_img is None:
#            raise RuntimeError("Image file %s does not exist or is not a valid"\
#                    " image")
#        self._img_ht,self._img_wd = self._data_img.shape[:2]
#        print("...Done.  Shape is %s" % repr(self._data_img.shape))


# 224x224 patches to feed into pre-trained ResNet, classifying center pixel
class RGBPatchesCenter(Dataset):
    def __init__(self, data_dir_or_file, labels_dir_or_file, cats_dict={},
            upsample=2, ingest_patch_size=224):
        self._cats_dict = cats_dict
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
        patch = np.transpose(patch, (2,0,1))
        label = self._labels_img[i_beg][j_beg]
        label = 0 if label not in self._cats_dict \
                else self._cats_dict[label]
        label = torch.LongTensor([label]).squeeze()
        return patch,label

    def __len__(self):
        return self._img_ht * self._img_wd

    def get_cats_dict(self):
        return self._cats_dict

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
class TBChips(CropNetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, index):
        chunk,_ = self._get_chunk_and_label(index)
        index = index // self.num_chunks()
        ht,wd = chunk.shape[2:]
        i = index // ht
        j = index % ht
        tb_chip = np.squeeze( chunk[:,:,i,j] )
        tb_chip = torch.FloatTensor(tb_chip).unsqueeze(0)
        self._update_item_ct()
        return tb_chip,tb_chip

    def get_data(self):
        return self._tb_chips

    def get_image_bbox(self):
        x0,y0 = self._src_image_x,self._src_image_y
        x1,y1 = x0+self._src_image_size,y0+self._src_image_size
        return (x0,y0,x1,y1)

    def _calc_N(self):
        acc = 0
        for p in self._data_paths:
            bbox = get_bbox_from_file_path(p)
            acc += bbox_area(bbox)
        self._N = int(acc)

    def _load_tb_chips_from_npy(self, data_file):
        full_tb_chips = np.load(data_file)
        x0,y0,x1,y1 = self.get_image_bbox()
        self._tb_chips = full_tb_chips[:, :, y0:y1, x0:x1]
        self._N = self._tb_chips.shape[2] * self._tb_chips.shape[3]



def _test_normalize_feats(features, do_copy=False):
    if len(features.shape) != 3:
        raise RuntimeError("Expecting features input to have 3 dimensions")
    if do_copy:
        features = np.copy(features)
    for k in range(features.shape[2]):
        c = features[:,:,k]
        features[:,:,k] = (c - np.min(c)) / (np.max(c) - np.min(c))
    return features

def _test_main(args):
    if not pe(args.output_supdir):
        os.makedirs(args.output_supdir)
    output_dir = pj(args.output_supdir, args.test)
    if not pe(output_dir):
        os.makedirs(output_dir)

    if args.test == "RGBPatches":
        print("Testing RGBPatches...")
        dataset = RGBPatches(data_dir=args.data_dir, labels_dir=args.labels_dir,
                mode="train")
        num_cats = dataset.get_num_cats()
        inc = 256 // num_cats
        print("The length of %s is %d." % (args.test, len(dataset)))
        N = len(dataset) if args.num_outputs<0 else args.num_outputs
        for i in range(N):
            patch,label = dataset[i % len(dataset)]
            patch = np.transpose( patch.cpu().data.numpy(), (1,2,0) )
            label = label.cpu().data.numpy() * inc
            print("Patch %i range, (%f, %f); label %i range, (%i, %i)" \
                    % (i, np.min(patch), np.max(patch), i, np.min(label),
                        np.max(label)))
            print("\tShape: %s" % repr(patch.shape))
            patch = _test_normalize_feats(patch, do_copy=True)
            patch = (patch * 255.0).astype(np.uint8)
            cv2.imwrite(pj(output_dir, "patch_%03d.png" % (i)), patch)
            cv2.imwrite(pj(output_dir, "label_%03d.png" % (i)), label)
        print("...Done")
    elif args.test == "TBChips":
        print("Testing TBChips...")
        tiles_per_cohort = 2
        print("Tiles per cohort: %d" % (tiles_per_cohort))
        dataset = TBChips(data_dir=args.data_dir_or_file, labels_dir=None,
                tiles_per_cohort=tiles_per_cohort)
        N = len(dataset)
        print("The length of %s is %d." % (args.test, N))
        output_dir = pj(args.output_supdir, args.test)
        print("Test chips written to %s" % (output_dir))
        if not pe(output_dir):
            os.makedirs(output_dir)
        inc = N // args.num_outputs
        for i in range(args.num_outputs):
            tb_chip,_ = dataset[i*inc]
            tb_chip = tb_chip.cpu().data.numpy()
            print("Writing chip %d, (shape, min, max): %s, %f, %f" \
                    % (i, tb_chip.shape, np.min(tb_chip), np.max(tb_chip)))
            tb_chip = np.transpose(tb_chip*255.0, (1,2,0) )
            chip_path = pj(output_dir, "tb_chip_%03d.png" % (i))
            cv2.imwrite(chip_path, tb_chip)
        print("...Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", type=str, default="RGBPatches",
            choices=["RGBPatches", "TBChips"])
    parser.add_argument("-d", "--data-dir", type=str,
            default=pj(HOME, "Training/cropnet/sessions/session_07/feats_maps"))
    parser.add_argument("-l", "--labels-dir", type=str,
            default=pj(DATA, "Datasets/HLS/tb_data/train/cdl"))
    parser.add_argument("-o", "--output-supdir", type=str, 
            default=pj(HOME, "Training/cropnet/test_out"))
    parser.add_argument("-n", "--num-outputs", type=int, default=20)
    args = parser.parse_args()
    _test_main(args)
    
