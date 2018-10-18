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
from datasets import TBChips
from models import CropNetFCAE
from pyt_utils.trainer import TrainerBase, create_session_dir, ae_sampler
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torchvision.utils import save_image


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

def get_model(chip_size, bneck_size):
    model = CropNetFCAE(chip_size, bneck_size)
    return model.cuda()

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
#    assert (recon_x >= 0. & recon_x <= 1.).all()
#    assert (x >= 0. & x <= 1.).all()
    recon_x = recon_x.view(-1, size_sq)
    x = x.view(-1, size_sq)
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
#    e12 = torch.FloatTensor([1e-12]).cuda()
#    white = torch.sum((1-x) * torch.log( torch.max(1-recon_x,e12)))
#    black = torch.sum(x * torch.log( torch.max(recon_x,e12)) )
#    print(white, black)
#    print("BCE", white+black, BCE)


    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return (BCE + KLD, BCE, KLD)


def main(args):
    session_dir = create_session_dir(args.output_supdir)
    train_loader = get_data_loader(args.data_dir, args.src_image_x, 
            args.src_image_y, args.src_image_size, args.batch_size)
    model = get_model(19, 3) # TODO
    opt_getter = lambda lr : get_optimizer(model, args.opt_name, lr)
    criterion = lambda output,x : loss_function(output, x, 19) # TODO
    trainer = TrainerBase(model,
            (train_loader, None),
            opt_getter=opt_getter,
            criterion=criterion,
            session_dir=session_dir,
            epoch_writer=ae_sampler,
            batch_writer=batch_writer
            )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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

