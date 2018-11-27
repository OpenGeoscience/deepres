"""
UNet-like model for doing segmentation, borrowing liberally from TernausNet
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


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

class Upsample(nn.Module):
    def __init__(self, scale_factor, mode, **kwargs):
        super().__init__(**kwargs)
        self._scale_factor = scale_factor
        self._mode = mode

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self._scale_factor,
                mode=self._mode)

# Courtesy of https://github.com/ternaus/TernausNetV2.git 
class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

# Courtesy of https://github.com/ternaus/TernausNetV2.git 
class DecoderBlock(nn.Module):
    """Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, middle_channels, out_channels, 
            is_deconv=False, scale_factor=2):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels
        self.middle_channels = middle_channels
        self.out_channels = out_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4,
                    stride=2, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                Upsample(scale_factor=scale_factor, mode="nearest"),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels)
            )

    def forward(self, x):
        return self.block(x)


class CropUNet(nn.Module):
    def __init__(self, backbone="resnet18", num_classes=4):
        super().__init__()
        self._decoder = None
        self._encoder = None
        self._in_shapes = None
        self._num_classes = num_classes
        self._out_shapes = None
        self._resnet = None

        if backbone == "resnet18":
            self._resnet = resnet18(pretrained=True)
        elif backbone == "resnet34":
            self._resnet = resnet34(pretrained=True)
        elif backbone == "resnet50":
            self._resnet = resnet50(pretrained=True)
        else:
            raise RuntimeError("Unrecognized backbone network, %s" % backbone)
        self._encoder = OrderedDict()
        for name,module in self._resnet.named_children():
            if name == "fc": break
            self._encoder[name] = module

        self._calc_layer_shapes()
        
        is_deconv = False # TernausNetV2 default
        num_filters = 64 #32
        center_sz = self._out_shapes[-1][0]
        self.center = DecoderBlock(center_sz, num_filters*8, num_filters*8,
                is_deconv=is_deconv, scale_factor=7)
        self.dec5 = DecoderBlock(center_sz + num_filters*8, num_filters*8,
                num_filters*8, is_deconv=is_deconv)
        self.dec4 = DecoderBlock(center_sz//2 + num_filters*8, num_filters*8,
                num_filters*8, is_deconv=is_deconv)
        self.dec3 = DecoderBlock(center_sz//4 + num_filters*8, num_filters*2,
                num_filters*2, is_deconv=is_deconv)
        self.dec2 = DecoderBlock(center_sz//8 + num_filters*2, num_filters*2,
                num_filters, is_deconv=is_deconv)
        self.dec1 = ConvRelu(center_sz//8 + num_filters, num_filters)
#        self.dec1 = ConvRelu(center_sz//16 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
        self._decoder = OrderedDict()
        self._decoder["center"] = self.center
        self._decoder["dec5"] = self.dec5
        self._decoder["dec4"] = self.dec4
        self._decoder["dec3"] = self.dec3
        self._decoder["dec2"] = self.dec2
        self._decoder["dec1"] = self.dec1
        self._decoder["final"] = self.final

    def forward(self, x):
        conv1 = self._encoder["conv1"](x)
        bn1 = self._encoder["bn1"](conv1)
        relu = self._encoder["relu"](bn1)
        maxpool = self._encoder["maxpool"](relu)
        layer1 = self._encoder["layer1"](maxpool)
        layer2 = self._encoder["layer2"](layer1)
        layer3 = self._encoder["layer3"](layer2)
        layer4 = self._encoder["layer4"](layer3)
        avgpool = self._encoder["avgpool"](layer4)

        print("Avgpool shape: %s" % repr(avgpool.shape))
#        center = self.center(self.pool(avgpool))
        center = self.center(layer4)

        dec5 = self.dec5(torch.cat([center, layer4], 1))
        dec4 = self.dec4(torch.cat([dec5, layer3], 1))
        dec3 = self.dec3(torch.cat([dec4, layer2], 1))
        dec2 = self.dec2(torch.cat([dec3, layer1], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.final(dec1)

    def _calc_layer_shapes(self):
        x = torch.FloatTensor(1,3,224,224)
        self._in_shapes = []
        self._out_shapes = []
        for _,module in self._encoder.items():
            self._in_shapes.append(x.shape[1:])
            x = module(x)
            self._out_shapes.append(x.shape[1:])
        
def _test_show_layers(tnet):
    x = torch.FloatTensor(1,3,224,224)
    convs = []
    for name,m in tnet._encoder.items():
        if name=="avgpool": break
        in_shape = x.shape
        x = m(x)
        if name=="relu" or name.startswith("layer"):
            convs.append( x.clone() )
        out_shape = x.shape
        print("%s, In: %s, Out: %s" % (name, repr(in_shape), repr(out_shape)))
    convs = convs[::-1]
    deconv_mods = [tnet.dec5,tnet.dec4,tnet.dec3,tnet.dec2,tnet.dec1,tnet.final]
    for i,m in enumerate(deconv_mods):
        in_shape = x.shape
        if i==5:
            x = m(x)
            name = "final"
        else:
#            print(x.shape, torch.cat([x, convs[i]], 1).shape)
#            print(m.in_channels)
            x = m( torch.cat([x, convs[i]], 1) )
            name = "dec%d" % (5-i)
        print("%s, In: %s, Out: %s" % (name, repr(in_shape), repr(out_shape)))


def _main(args):
    if not pe(args.output_supdir):
        os.makedirs(args.output_supdir)
    output_dir = pj(args.output_supdir, "cropunet")
    if not pe(output_dir):
        os.makedirs(output_dir)
    
    crop_unet = CropUNet(backbone=args.model_name)
    with open(pj(output_dir, "crop_unet_model.txt"), "w") as fp:
        fp.write(repr(crop_unet) + "\n")

    _test_show_layers(crop_unet)
    raise

    sz = (3, 224, 224)
    x = torch.FloatTensor(1, *sz)
    specs = OrderedDict()
    for m in crop_unet._resnet.named_children():
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

    xd = torch.FloatTensor(1, 512+256, 256, 256)
    for name in ["dec5","dec4","dec3","dec2", "dec1","final"]:
        m = crop_unet._decoder[name]
        in_shape = xd.shape
        xd = m(xd)
        out_shape = xd.shape
        print("%s, In: %s, Out: %s" % (name, repr(list(in_shape)),
            repr(list(out_shape))))

    with open(pj(output_dir, "crop_unet_model.txt"), "w") as fp:
        json.dump(specs, fp, indent=2)

    x = torch.FloatTensor(1, *sz)
    y = crop_unet(x)
    print(y.shape, type(y))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-supdir", type=str,
            default=pj(HOME, "Training/cropnet/test_out"))
    parser.add_argument("-m", "--model-name", type=str, default="resnet18",
            choices=["resnet18", "resnet34", "resnet50"])
    args = parser.parse_args()
    _main(args)


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

