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
from datasets import RGBPatches, RGBPatches2
from cropunet import CropUNet
from utils import make_clut, transform_cdl, get_cat_list

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")
if platform.node() == "matt-XPS-8900":
    DATA = HOME
else:
    DATA = "/media/data"


def get_loaders(cfg):
    cats = get_cat_list(cfg["cats"])
    train_dataset = RGBPatches(data_dir=cfg["data_dir"],
            labels_dir=cfg["labels_dir"],
            cats=cats,
            mode="train")
    train_loader = DataLoader(train_dataset,
            batch_size=args.batch_size,
            num_workers=cfg["num_workers"],
            shuffle=True)
    test_dataset = RGBPatches(data_dir=cfg["data_dir"],
            labels_dir=cfg["labels_dir"],
            cats=cats,
            mode="test")
    test_loader = DataLoader(test_dataset,
            batch_size=args.batch_size,
            num_workers=cfg["num_workers"],
            shuffle=False)
    return train_loader,test_loader

def main(args):
    cfg = vars(args)
    train_loader,test_loader = get_loaders(cfg)
    cats = get_cat_list(cfg["cats"])
    num_classes = len(cats) # if cfg["cats"]=="all" else len(cats)+1 TODO!
    model = CropUNet(num_classes=len(cats))
    if args.use_cuda:
        model = model.cuda()
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay)
    sen = cfg["save_every_N"]
    for epoch in range(args.num_epochs):
        iterator = iter(train_loader)
        for b in range(len(train_loader.dataset) // args.batch_size):
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
#            print(patches.shape, yhat.shape)
#            print(torch.min(patches), torch.max(patches), torch.median(patches))
#            print("\t", torch.min(yhat), torch.max(yhat), torch.median(yhat))
            print(yhat.shape, labels.shape)
#            raise
            loss = criterion(yhat, labels)
Need to zero out/replace with null value all unused categories
            loss.backward()
            optimizer.step()
            preds = torch.argmax(yhat, dim=1)
#            print(torch.min(yhat), torch.max(yhat), torch.median(yhat))
#            print(np.unique( preds.cpu().data.numpy() ))
            acc = 100.0 * np.mean( (preds==labels).cpu().data.numpy() )

        new_labels = []
        new_preds = []
        for pred,label in zip(preds,labels):
            label,_ = transform_cdl(label.cpu().data.numpy())
            label = np.transpose(label, (2,0,1))
            label = torch.cuda.FloatTensor(label)
            new_labels.append(label)
            pred,_ = transform_cdl(pred.cpu().data.numpy())
            pred = np.transpose(pred, (2,0,1))
            pred = torch.cuda.FloatTensor(pred)
            new_preds.append(pred)

        print("Epoch %d: Loss %0.4f, Acc. %0.2f" % (epoch, loss.item(), acc))
        if epoch % sen == sen-1:
            for i,(patch,pred,label) in enumerate( zip(patches, new_preds,
                new_labels) ):
                tv.utils.save_image([patch, pred, label],
                        pj(os.path.dirname(os.path.abspath(args.output_dir)),
                            "samples/cropunet/segments_%03d_%03d.png" \
                                    % (epoch, i)))

            iterator = iter(test_loader)
            for b in range(len(test_loader.dataset) // args.batch_size):
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
#                print(patches.shape, yhat.shape)
#                print(torch.min(patches), torch.max(patches),
#                   torch.median(patches))
#                print("\t", torch.min(yhat), torch.max(yhat),
#                    torch.median(yhat))
                loss = criterion(yhat, labels)
                preds = torch.argmax(yhat, dim=1)
#                print(torch.min(yhat), torch.max(yhat), torch.median(yhat))
#                print(np.unique( preds.cpu().data.numpy() ))
                acc = 100.0 * np.mean( (preds==labels).cpu().data.numpy() )

            s = "TEST: Loss %0.4f, Acc. %0.2f" % (loss.item(), acc)
            print(s)
            with open(pj(args.output_dir, "session.log"), "a") as fp:
                fp.write(s + "\n")
    
            new_labels = []
            new_preds = []
            for pred,label in zip(preds,labels):
                label,_ = transform_cdl(label.cpu().data.numpy())
                label = np.transpose(label, (2,0,1))
                label = torch.cuda.FloatTensor(label)
                new_labels.append(label)
                pred,_ = transform_cdl(pred.cpu().data.numpy())
                pred = np.transpose(pred, (2,0,1))
                pred = torch.cuda.FloatTensor(pred)
                new_preds.append(pred)
            for i,(patch,pred,label) in enumerate( zip(patches, new_preds,
                new_labels) ):
                tv.utils.save_image([patch, pred, label],
                        pj(os.path.dirname(os.path.abspath(args.output_dir)),
                            "samples/cropunet/segments_test_%03d_%03d.png" \
                                    % (epoch, i)))
            torch.save(model.state_dict(), pj(args.output_dir,
                "CropUNet_%04d.pkl" % (epoch)))
    

def _test_main(args):
    cats_dict = make_clut()
    cfg = vars(args)
    dataset = RGBPatches(data_dir=args.data_dir_or_file,
            labels_dir=args.labels_dir_or_file,
            cats=get_cats(cfg),
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
    parser.add_argument("-d", "--data-dir", type=str,
            default=pj(HOME, "Training/cropnet/sessions/session_07/feats.npy"))
    parser.add_argument("-l", "--labels-dir", type=str,
            default=pj(DATA, "Datasets/HLS/test_imgs/cdl/" \
                    "cdl_2016_neAR_0_0_500_500.npy"))
    parser.add_argument("-o", "--output-dir", type=str,
            default=pj(HOME, "Training/cropnet/models"))
    parser.add_argument("-s", "--image-size", type=int, default=256)
    parser.add_argument("-n", "--num-epochs", type=int, default=100)
    parser.add_argument("-b", "--batch-size", type=int, default=4) # TODO
    parser.add_argument("--cats", "--categories", dest="cats", type=str,
            default="original4",
            choices=["original4", "top5", "top10", "top25", "top50", "all"])
    parser.add_argument("--lr", "--learning-rate", dest="lr", type=float,
            default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--no-cuda", dest="use_cuda", action="store_false")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--save-every-N", type=int, default=10,
            help="Measured in epochs")
    args = parser.parse_args()
    if args.test:
        _test_main(args)
    else:
        main(args)

