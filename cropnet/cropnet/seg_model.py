"""
The image segmentation model for pixel classification

To get *simple* pseudo RGB images from the data, just take R, G, B bands from
the spectral data!
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

from unet import DynamicUnet
from torch.autograd import Variable

from torchvision.models import resnet18, resnet34, resnet50

def conv_block(c_in, c_out, ksz=3, stride=2, padding=1):
    block = nn.Sequential([
        nn.Conv2d(c_in, c_out, kernel_size=ksz, stride=stride, padding=padding),
        nn.ReLu(),
        nn.BatchNorm2d(c_out)
        ])

class CropSeg(nn.Module):
    def __init__(self, image_size, num_classes):
        super(CropSeg, self).__init__()
        self._name = "cropseg"

        if type(image_size) is not tuple or len(image_size)!=3:
            raise RuntimeError("image_size parameter must be a HxWxC tuple")
           
        img_ht = image_size.shape(0)
        img_wd = image_size.shape(1)
        img_c = image_size.shape(2)
        conv1 = conv_block(img_c, img_c*2)
        conv2 = conv_block(img_c*2, img_c*4)
        conv3 = conv_block(img_c*4, img_c*8)
        conv4 = conv_block(img_c*8, img_c*16)
        conv5 = conv_block(img_c*16, img_c*32)

        self._encoder = nn.Sequential([conv1, conv2, conv3, conv4, conv5])

        self._model = DynamicUNet(self._encoder)

    def forward(self, x):
        return self._model(x)

    def get_name(self):
        return self._name

class Pretrained(nn.Module):
    def __init__(self, model_name="resnet18", num_cats=10):
        super(Pretrained, self).__init__()
        self._model = None
        self._name = "pretrained"
        self._num_cats = num_cats

        if model_name=="resnet18":
            self._model = resnet18(pretrained=True)
        elif model_name=="resnet34":
            self._model = resnet34(pretrained=True)
        elif model_name=="resnet50":
            self._model = resnet50(pretrained=True)
        else:
            raise RuntimeError("%s is not a recognized model" % (model_name))

        num_features = self._model.fc.in_features
        self._model.fc = nn.Linear(num_features, num_cats)

    def forward(self, x):
        return self._model(x)

    def get_name(self):
        return self._name

