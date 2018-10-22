"""
This file trains the cropnet models, either the AE or classifier depending on choice.
"""

import argparse
import csv
import logging
import numpy as np
import os
import shutil
import torch
import torch.nn.functional as F

from collections import OrderedDict

# pytorch imports
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# ml_utils imports
from pyt_utils.trainer import TrainerBase, create_session_dir, ae_sampler

# Local imports
from datasets import TBChips
from models import CropNetFCAE
from utils import get_features, load_model

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

def get_data_loader(data_dir, image_x, image_y, image_size, batch_size):
    dataset = TBChips(data_dir, image_x, image_y, image_size)
    train_loader = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0) # TODO
    return train_loader # TODO test loader!!

def get_optimizer(model, opt_name, lr):
    if opt_name=="SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_name=="Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise RuntimeError("Unrecognized optimizer, %s" % (opt_name))
    return optimizer

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


def main(args):
    session_dir = create_session_dir(args.output_supdir)
    test_loader = None
    ae_model = None
    if args.ae_model is None:
        train_loader = get_data_loader(args.data_dir, args.src_image_x, 
                args.src_image_y, args.src_image_size, args.batch_size)
        test_loader = train_loader
        ae_model = make_ae_model(args.network, 19, 3) # TODO
        opt_getter = lambda lr : get_optimizer(ae_model, args.opt_name, lr)
        criterion = lambda output,x : loss_function(output, x, 19) # TODO
        ae_trainer = TrainerBase(ae_model,
                (train_loader, None),
                opt_getter=opt_getter,
                criterion=criterion,
                session_dir=session_dir,
                epoch_writer=ae_sampler,
                batch_writer=batch_writer
                )
        ae_trainer.train()
    seg_model = make_seg_model(args.network, 256) # TODO
    if ae_model is None:
        ae_model = load_ae_model(args.ae_model_path, args.network, chip_size=19,
                bneck_size=3) # TODO
    if test_loader is None:
        test_loader = get_data_loader(args.data_dir, args.src_image_x, 
                args.src_image_y, args.src_image_size, args.batch_size)
    features = get_features(ae_model, test_loader) # TODO this should operate 
        # over an entire directory
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--ae-model-path", type=str, default=None,
            help="Optionally supply a pre-trained AE model, otherwise AE will" \
                    " be retrained from scratch")
    parser.add_argument("-d", "--data-dir", type=str,
            default=pj(HOME, "Datasets/HLS/test_imgs/hls"))
    parser.add_argument("-o", "--output-supdir", type=str,
            default=pj(HOME, "Training/cropnet/sessions"))
    parser.add_argument("--src-image-x", type=int, default=0,
            help="Source chip top coordinate")
    parser.add_argument("--src-image-y", type=int, default=0,
            help="Source chip left coordinate")
    parser.add_argument("--src-image-size", type=int, default=500)
    parser.add_argument("--network", type=str, default="CropNetFCAE",
            choices=["CropNetFCAE", "CropSeg"])

    parser.add_argument("--opt-name", type=str, default="Adam",
            choices=["Adam", "SGD"])
    parser.add_argument("--batch-size", type=int, default=64)
    
    args = parser.parse_args()
    main(args)

