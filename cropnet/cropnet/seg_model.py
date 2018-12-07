"""
The image segmentation model for pixel classification

To get *simple* pseudo RGB images from the data, just take R, G, B bands from
the spectral data!
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
from torchvision.models import vgg16

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")

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

def _main(args):
    if not pe(args.output_supdir):
        os.makedirs(args.output_supdir)
    output_dir = pj(args.output_supdir, "CropSeg")
    if not pe(output_dir):
        os.makedirs(output_dir)

    sz = (3, args.image_size, args.image_size)
    cropseg = CropSeg(sz, args.num_classes)
    cropset_str = repr(cropseg)
    with open(pj(output_dir, "cropseg_model.txt"), "w") as fp:
        fp.write(cropset_str + "\n")
    if args.use_cuda:
        cropseg = cropseg.cuda()
    x = torch.rand( (1,) + sz )
    if args.use_cuda:
        x = x.cuda()
    preds = cropseg(x)
    print("Output shape: %s" % repr(preds.shape))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--image-size", type=int, default=224)
    parser.add_argument("--nc", "--num-classes", dest="num_classes", type=int,
            default=4)
    parser.add_argument("-o", "--output-supdir", type=str, 
            default=pj(HOME, "Training/cropnet/test_out"))
    parser.add_argument("--no-cuda", dest="use_cuda", action="store_false")
    args = parser.parse_args()
    _main(args)

