"""
This file trains the cropnet models, either the AE or classifier depending on 
choice.
"""

import argparse
import csv
import logging
import numpy as np
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

# pytorch imports
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# ml_utils imports
from general.utils import create_session_dir

# Local imports
from ae_model import CropNetFCAE, load_ae_model
from ae_trainer import AETrainer
from datasets import RGBPatchesCenter, TBChips
from seg_model import CropSeg, Pretrained
from utils import get_features

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


g_session_log = "session.log"

def batch_writer(trainer, epoch, batch_idx, batch_len, losses):
    loss = losses[0].item()
    BCE = losses[1].item()
    KLD = losses[2].item()
    chip_sz = trainer.get_model().get_input_size()
    loss_div = batch_len * chip_sz * chip_sz
    s = "\t\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tBCE: {:.6f}" \
            "\tKLD: {:.6f}\n".format(epoch,
                batch_idx * batch_len,
                len(trainer.get_train_loader().dataset),
                100.0 * batch_idx / len(trainer.get_train_loader()),
                loss / loss_div, BCE / loss_div, KLD / loss_div)
    with open(pj(trainer.get_session_dir(), g_session_log), "a") as fp:
        fp.write(s)
        fp.flush

def get_cats(cdl_file_path):
    cdl = np.load(cdl_file_path)
    THRESH = 0.05 # TODO
    cats = []
    for i in range(0, 256):
        if np.sum(i==cdl) / cdl.size > THRESH:
            cats.append(i)
    return cats

def get_data_loader(data_dir, image_x, image_y, image_size, batch_size):
    dataset = TBChips(data_dir, image_x, image_y, image_size)
    train_loader = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=8)
    return train_loader # TODO test loader!!

def get_optimizer(model, opt_name, lr):
    if opt_name=="SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_name=="Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise RuntimeError("Unrecognized optimizer, %s" % (opt_name))
    return optimizer

def get_seg_loader(feats_file_path, cdl_file_path, batch_size, cats):
    cats_dict = OrderedDict()
    for i,c in enumerate(cats):
        cats_dict[c] = i+1 # Reserve 0 for background category
    dataset = RGBPatchesCenter(feats_file_path, cdl_file_path, cats_dict)
    train_loader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=8)
    return train_loader # TODO test loader


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(model_output, x, chip_size):
    recon_x,mu,logvar = model_output
    size_sq = chip_size*chip_size
    recon_x = recon_x.view(-1, size_sq)
    x = x.view(-1, size_sq)
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return (BCE + KLD, BCE, KLD)

def make_ae_model(ae_network, chip_size, bneck_size):
    if ae_network == "CropNetFCAE":
        model = CropNetFCAE(chip_size, bneck_size)
    else:
        raise RuntimeError("Unrecognized network %s" % (ae_network))
    return model.cuda()

def make_seg_model(seg_network, **kwargs):
    if seg_network == "Pretrained":
        model = Pretrained(**kwargs)
    elif seg_network == "CropSeg":
        raise NotImplementedError()
    else:
        raise RuntimeError("Model %s not recognized" % (seg_network))
    return model.cuda().eval()

#    def __init__(self, model, loaders, criterion, session_dir, num_epochs=1000,              base_lr=0.001, num_lr_drops=2, lr_drop_factor=5, log_interval=10):

def main(args):
    session_dir = create_session_dir(args.output_supdir)
    test_ae_loader = None
    ae_model = None
    if args.ae_model_path is None:
        train_ae_loader = get_data_loader(args.data_dir, args.src_image_x, 
                args.src_image_y, args.src_image_size, args.batch_size)
        test_ae_loader = get_data_loader(args.test_data_dir, args.test_image_x,
                args.test_image_y, args.test_image_size, args.batch_size)
        ae_model = make_ae_model(args.network, 19, 3) # TODO
        ae_trainer = AETrainer(
                model=ae_model,
                loaders=(train_ae_loader, test_ae_loader),
                session_dir=session_dir)
        ae_trainer.train()

    if args.ae_model_only:
        cats = get_cats(args.cdl_file_path)
        num_cats = len(cats)
        print("Number of land cover categories above threshold: %d" \
                % (num_cats))
        seg_model = make_seg_model("Pretrained", model_name="resnet18", 
                num_cats=num_cats+1) # TODO
        if ae_model is None:
            ae_model = load_ae_model(args.ae_model_path, args.network, 
                    chip_size=19, bneck_size=3) # TODO
        if test_ae_loader is None:
            test_ae_loader = get_data_loader(args.data_dir, args.src_image_x, 
                    args.src_image_y, args.src_image_size, args.batch_size)
        features = get_features(ae_model, test_ae_loader) # TODO this should 
            # operate over an entire directory
        feats_path = pj(session_dir, "feats.npy")
        np.save(feats_path, features)
        train_seg_loader = get_seg_loader(feats_path, args.cdl_file_path,
                args.batch_size, cats)
        criterion = nn.CrossEntropyLoss()
        seg_trainer = ClassTrainer(seg_model,
                (train_seg_loader, None),
                criterion=criterion,
                session_dir=session_dir)
        seg_trainer.train()      


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--ae-model-path", type=str, default=None,
            help="Optionally supply a pre-trained AE model, otherwise AE will" \
                    " be retrained from scratch")
    parser.add_argument("-d", "--data-dir", type=str,
            default=pj(HOME, "Datasets/HLS/test_imgs/hls"))
    parser.add_argument("--test-data-dir", type=str,
            default=pj(HOME, "Datasets/HLS/test_imgs/hls"))
    parser.add_argument("--cdl-file-path", type=str,
            default=pj(HOME, "Datasets/HLS/test_imgs/cdl/" \
                    "cdl_2016_neAR_0_0_500_500.npy")) # TODO
    parser.add_argument("-o", "--output-supdir", type=str,
            default=pj(HOME, "Training/cropnet/sessions"))

    parser.add_argument("--src-image-x", type=int, default=0,
            help="Source chip top coordinate")
    parser.add_argument("--src-image-y", type=int, default=0,
            help="Source chip left coordinate")
    parser.add_argument("--src-image-size", type=int, default=500)
    parser.add_argument("--network", type=str, default="CropNetFCAE",
            choices=["CropNetFCAE", "CropSeg", "Pretrained"])

    parser.add_argument("--test-image-x", type=int, default=500,
            help="Test chip top coordinate")
    parser.add_argument("--test-image-y", type=int, default=0,
            help="Test chip left coordinate")
    parser.add_argument("--test-image-size", type=int, default=500)

    parser.add_argument("--opt-name", type=str, default="Adam",
            choices=["Adam", "SGD"])
    parser.add_argument("--batch-size", type=int, default=64)
    
    args = parser.parse_args()
    main(args)

