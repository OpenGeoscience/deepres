"""
This file holds the trainer for the segmentation network.
"""

import argparse
import numpy as np
import os
import platform

# pytorch includes
import torch
import torch.nn as nn
import torchvision as tv
from torch.autograd import Variable
from torch.utils.data import DataLoader

# local includes
from datasets import RGBPatches
from cropunet import CropUNet
from utils import make_clut

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")
if platform.node() == "matt-XPS-8900":
    DATA = HOME
else:
    DATA = "/media/data"


def main(args):
    train_dataset = RGBPatches(args.data_dir_or_file, args.labels_dir_or_file,
            mode="train")
    train_loader = DataLoader(train_dataset,
            batch_size=args.batch_size,
            num_workers=8,
            shuffle=True)
    model = CropUNet(num_classes=args.num_classes)
    if args.use_cuda:
        model = model.cuda()
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay)
    for epoch in range(args.num_epochs):
        iterator = iter(train_loader)
        for b in range(len(train_dataset) // args.batch_size):
            patches,labels = next(iterator)
            if args.use_cuda:
                patches = Variable(patches).cuda()
                with torch.no_grad():
                    labels = Variable(labels).cuda()
            else:
                patches = Variable(patches)
                with torch.no_grad():
                    labels = Variable(labels)
            optimizer.zero_grad()
            yhat = model(patches)
            loss = criterion(yhat, labels)
            loss.backward()
            optimizer.step()
            yhat = yhat.cpu().detach().numpy()
            preds = np.argmax(yhat, axis=1)
            acc = 100.0 * np.mean( preds==labels.cpu().data.numpy() )
        print("Epoch %d: Loss %0.4f, Acc. %0.2f" % (epoch, loss.item(), acc))
    

def _test_main(args):
    cats_dict = make_clut()
    dataset = RGBPatches(args.data_dir_or_file, args.labels_dir_or_file,
            mode="train", cats_dict=cats_dict)
    data_loader = DataLoader(dataset,
            batch_size=args.batch_size,
            num_workers=1,
            shuffle=False)
    model = CropUNet(num_classes=args.num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay)
    iterator = iter(data_loader)
    patches,labels = next(iterator)
    yhat = model(patches)
    print("patches type: %s, %s" % (type(patches), patches.dtype))
    print("labels type: %s, %s" % (type(labels), labels.dtype))
    print("yhat type: %s, %s" % (type(yhat), yhat.dtype))
    loss = criterion(yhat, labels)
    print("Loss: %f" % (loss.item()))
    loss.backward()
    optimizer.step()
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
            help="If set, just run the test code")
    parser.add_argument("--ae-model-path", type=str, 
            default=pj(HOME, "Training/cropnet/sessions/session_10/models/" \
                    "pretrained.pkl"))
    parser.add_argument("-d", "--data-dir-or-file", type=str,
            default=pj(HOME, "Training/cropnet/sessions/session_07/feats.npy"))
    parser.add_argument("-l", "--labels-dir-or-file", type=str,
            default=pj(DATA, "Datasets/HLS/test_imgs/cdl/" \
                    "cdl_2016_neAR_0_0_500_500.npy"))
    parser.add_argument("-s", "--image-size", type=int, default=256)
    parser.add_argument("-n", "--num-epochs", type=int, default=100)
    parser.add_argument("-b", "--batch-size", type=int, default=4) # TODO
    parser.add_argument("--nc", "--num-classes", dest="num_classes", type=int,
            default=5)
    parser.add_argument("--lr", "--learning-rate", dest="lr", type=float,
            default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--no-cuda", dest="use_cuda", action="store_false")
    args = parser.parse_args()
    if args.test:
        _test_main(args)
    else:
        main(args)

