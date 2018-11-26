"""
UNet-like model for doing segmentation
"""

import argparse
import json
import numpy as np
import os
from collections import OrderedDict

# pytorch includes
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.autograd import Variable
from torchvision.models import resnet18, resnet34, resnet50

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


class CropUNet(nn.Module):
    def __init__(self, backbone="resnet18"):
        super().__init__()
        if backbone == "resnet18":
            resnet = resnet18(pretrained=True)
        elif backbone == "resnet34":
            resnet = resnet34(pretrained=True)
        elif backbone == "resnet50":
            resnet = resnet50(pretrained=True)
        else:
            raise RuntimeError("Unrecognized backbone network, %s" % backbone)
        self._encoder = []
        for name,layers in resnet.named_children():
            if name == "fc": break
            self._encoder.append(layers)

        self._deconv1 = DecoderBlock(specs["avgpool"]["out"], # TODO
        

def _main(args):
    if not pe(args.output_supdir):
        os.makedirs(args.output_supdir)
    output_dir = pj(args.output_supdir, "cropunet")
    if not pe(output_dir):
        os.makedirs(output_dir)
    
    crop_unet = CropUNet(backbone=args.model_name)
    with open(pj(output_dir, "crop_unet_model.txt"), "w") as fp:
        fp.write(repr(crop_unet) + "\n")

    sz = (3, 224, 224)
    x = torch.FloatTensor(1, *sz)
    specs = OrderedDict()
    for m in crop_unet._encoder.named_children():
        name,layers = m
        in_shape = x.shape
        if name == "fc":
            x = x.view(x.size(0), -1)
        x = layers(x)
        out_shape = x.shape
        print("%s, In: %s, Out: %s" % (name, repr(list(in_shape)),
            repr(list(out_shape))))
        specs[name] = { "in_shape" : in_shape, "out_shape" : out_shape }
        # TODO print out tensor input, output dimensions of each top-level block

    with open(pj(output_dir, "crop_unet_model.txt"), "w") as fp:
        json.dump(specs, fp, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-supdir", type=str,
            default=pj(HOME, "Training/cropnet/test_out"))
    parser.add_argument("-m", "--model-name", type=str, default="resnet18",
            choices=["resnet18", "resnet34", "resnet50"])
    args = parser.parse_args()
    _main(args)

#class ConvRelu(nn.Module):
#    def __init__(self, in_, out):
#        super(ConvRelu, self).__init__()
#        self.conv = conv3x3(in_, out)
#        self.activation = nn.ReLU(inplace=True)
#
#    def forward(self, x):
#        x = self.conv(x)
#        x = self.activation(x)
#        return x
#

#class DecoderBlock(nn.Module):
#    """Paramaters for Deconvolution were chosen to avoid artifacts, following
#    link https://distill.pub/2016/deconv-checkerboard/
#    """
#
#    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=False):
#        super(DecoderBlock, self).__init__()
#        self.in_channels = in_channels
#
#        if is_deconv:
#            self.block = nn.Sequential(
#                ConvRelu(in_channels, middle_channels),
#                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
#                                   padding=1),
#                nn.ReLU(inplace=True)
#            )
#        else:
#            self.block = nn.Sequential(
#                nn.Upsample(scale_factor=2, mode='nearest'),
#                ConvRelu(in_channels, middle_channels),
#                ConvRelu(middle_channels, out_channels)
#            )
#
#    def forward(self, x):
#        return self.block(x)
#

#        self.center = DecoderBlock(1024, num_filters * 8, num_filters * 8, is_deconv=is_deconv)
#        self.dec5 = DecoderBlock(1024 + num_filters * 8, num_filters * 8, num_filters * 8, is_deconv=is_deconv)
#        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8, num_filters * 8, is_deconv=is_deconv)
#        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 2, num_filters * 2, is_deconv=is_deconv)
#        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2, num_filters, is_deconv=is_deconv)
#        self.dec1 = ConvRelu(64 + num_filters, num_filters)
#        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
#
#    def forward(self, x):
#        conv1 = self.conv1(x)
#        conv2 = self.conv2(self.pool(conv1))
#        conv3 = self.conv3(self.pool(conv2))
#        conv4 = self.conv4(self.pool(conv3))
#        conv5 = self.conv5(self.pool(conv4))
#
#        center = self.center(self.pool(conv5))
#
#        dec5 = self.dec5(torch.cat([center, conv5], 1))
#
#        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
#        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
#        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
#        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
#        return self.final(dec1)
#

