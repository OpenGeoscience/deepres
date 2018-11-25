"""
UNet-like model for doing segmentation
"""

import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

from torch.autograd import Variable

from torchvision.models import resnet18, resnet34, resnet50

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


"""
ResNet18 top-level layers:
conv1
bn1
relu
maxpool
layer1
layer2
layer3
layer4
avgpool
fc
"""
class CropUNet(nn.Module):
    def __init__(self):
        self._encoder = resnet18(pretrained=True)
        

def _main(args):
    if not pe(args.output_supdir):
        os.makedirs(args.output_supdir)
    output_dir = pj(args.output_supdir, "cropunet")
    if not pe(output_dir):
        os.makedirs(output_dir)
    
    crop_unet = CropUNet()
    with open(pj(output_dir, "crop_unet_model.txt")) as fp:
        fp.write(repr(crop_unet) + "\n")

    for ch in self._encoder.children():
        pass
        # TODO print out tensor input, output dimensions of each top-level block

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-supdir", type=str,
            default=pj(HOME, "Training/cropnet/test_out"))
    parser.add_argument("-m", "--model-name", type=str, default="resnet18",
            choices=["resnet18", "resnet34", "resnet50"])
    args = parser.parse_args()
    _main(args)

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

