import os
import sys
import yaml
import time
import shutil
import torch
import visdom
import random
import argparse
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.utils import data
from tqdm import tqdm

import glob
import json
from ptsemseg.myutils.config import Config
from ptsemseg.myutils.utils import update_config
from ptsemseg.tasks.transforms import augment_flips_color, augment_multiple_operations
from ptsemseg.dataset.image_provider import ImageProvider
from ptsemseg.dataset.threeband_image import ThreebandImageType
from ptsemseg.dataset.multiband_image import MultibandImageType
from ptsemseg.dataset.neural_dataset import TrainDataset, ValDataset

from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader 
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer

from tensorboardX import SummaryWriter

def test(cfg, areaname):
    
    # Setup seeds
    torch.manual_seed(cfg.get('seed', 1337))
    torch.cuda.manual_seed(cfg.get('seed', 1337))
    np.random.seed(cfg.get('seed', 1337))
    random.seed(cfg.get('seed', 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Augmentations
    augmentations = cfg['training'].get('augmentations', None)
    data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
#    data_loader = get_loader(cfg['data']['dataset'])
#    data_path = cfg['data']['path']
#
#    t_loader = data_loader(
#        data_path,
#        is_transform=True,
#        split=cfg['data']['train_split'],
#        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
#        augmentations=data_aug)
#
#    v_loader = data_loader(
#        data_path,
#        is_transform=True,
#        split=cfg['data']['val_split'],
#        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),)
#
#    n_classes = t_loader.n_classes
#    trainloader = data.DataLoader(t_loader,
#                                  batch_size=cfg['training']['batch_size'], 
#                                  num_workers=cfg['training']['n_workers'], 
#                                  shuffle=True)
#
#    valloader = data.DataLoader(v_loader, 
#                                batch_size=cfg['training']['batch_size'], 
#                                num_workers=cfg['training']['n_workers'])
    datapath = '/home/chengjjang/Projects/deepres/SatelliteData/{}/'.format(areaname)
    paths = {
        'masks': '{}/patch{}_train/gt'.format(datapath, areaname),
        'images': '{}/patch{}_train/rgb'.format(datapath, areaname),
        'nirs': '{}/patch{}_train/nir'.format(datapath, areaname),
        'swirs': '{}/patch{}_train/swir'.format(datapath, areaname),
        'vhs': '{}/patch{}_train/vh'.format(datapath, areaname),
        'vvs': '{}/patch{}_train/vv'.format(datapath, areaname),
        'redes': '{}/patch{}_train/rede'.format(datapath, areaname),
        'ndvis': '{}/patch{}_train/ndvi'.format(datapath, areaname),
        }

    valpaths = {
        'masks': '{}/patch{}_train/gt'.format(datapath, areaname),
        'images': '{}/patch{}_train/rgb'.format(datapath, areaname),
        'nirs': '{}/patch{}_train/nir'.format(datapath, areaname),
        'swirs': '{}/patch{}_train/swir'.format(datapath, areaname),
        'vhs': '{}/patch{}_train/vh'.format(datapath, areaname),
        'vvs': '{}/patch{}_train/vv'.format(datapath, areaname),
        'redes': '{}/patch{}_train/rede'.format(datapath, areaname),
        'ndvis': '{}/patch{}_train/ndvi'.format(datapath, areaname),
        }
  
  
    n_classes = 3
    train_img_paths = [pth for pth in os.listdir(paths['images']) if ('_01_' not in pth) and ('_25_' not in pth)]
    val_img_paths = [pth for pth in os.listdir(valpaths['images']) if ('_01_' not in pth) and ('_25_' not in pth)]
    ntrain = len(train_img_paths)
    nval = len(val_img_paths)
    train_idx = [i for i in range(ntrain)]
    val_idx = [i for i in range(nval)]
    train_idx = [i for i in range(ntrain)]
    val_idx = [i for i in range(nval)]
    trainds = ImageProvider(MultibandImageType, paths, image_suffix='.png')
    valds = ImageProvider(MultibandImageType, valpaths, image_suffix='.png')

    print('valds.im_names: {}'.format(valds.im_names))
    
    config_path = 'crop_pspnet_config.json'
    with open(config_path, 'r') as f:
        mycfg = json.load(f)
        train_data_path = '{}/patch{}_train'.format(datapath, areaname)
        dataset_path, train_dir = os.path.split(train_data_path)
        mycfg['dataset_path'] = dataset_path
    config = Config(**mycfg)

    config = update_config(config, num_channels=12, nb_epoch=50)
    #dataset_train = TrainDataset(trainds, train_idx, config, transforms=augment_flips_color)
    dataset_train = TrainDataset(trainds, train_idx, config, 1)
    dataset_val = ValDataset(valds, val_idx, config, 1)
    trainloader = data.DataLoader(dataset_train,
                                  batch_size=cfg['training']['batch_size'], 
                                  num_workers=cfg['training']['n_workers'], 
                                  shuffle=True)

    valloader = data.DataLoader(dataset_val,
                                  batch_size=cfg['training']['batch_size'], 
                                  num_workers=cfg['training']['n_workers'], 
                                  shuffle=False)
    # Setup Metrics
    running_metrics_train = runningScore(n_classes)
    running_metrics_val = runningScore(n_classes)

    nbackground = 1116403140
    ncorn = 44080178
    nsoybean = 316698122

    print('nbackgraound: {}'.format(nbackground))
    print('ncorn: {}'.format(ncorn))
    print('nsoybean: {}'.format(nsoybean))
    
    wgts = [1.0, 1.0*nbackground/ncorn, 1.0*nbackground/nsoybean]
    total_wgts = sum(wgts)
    wgt_background = wgts[0]/total_wgts
    wgt_corn = wgts[1]/total_wgts
    wgt_soybean = wgts[2]/total_wgts
    weights = torch.autograd.Variable(torch.cuda.FloatTensor([wgt_background, wgt_corn, wgt_soybean]))

    # Setup Model
    model = get_model(cfg['model'], n_classes).to(device)

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k:v for k, v in cfg['training']['optimizer'].items() 
                        if k != 'name'}

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)

    scheduler = get_scheduler(optimizer, cfg['training']['lr_schedule'])

    loss_fn = get_loss_function(cfg)

    start_iter = 0
    runpath = '/home/chengjjang/arisia/CropPSPNet/runs/pspnet_crop_{}'.format(areaname)
    modelpath = glob.glob('{}/*/*_best_model.pkl'.format(runpath))[0]
    print('modelpath: {}'.format(modelpath))
    checkpoint = torch.load(modelpath)
    model.load_state_dict(checkpoint["model_state"])

    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    best_iou = -100.0

    respath = '{}_results_train'.format(areaname)
    os.makedirs(respath, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for inputdata in valloader:
            imname_val = inputdata['img_name']
            images_val = inputdata['img_data']
            labels_val = inputdata['seg_label']
            images_val = images_val.to(device)
            labels_val = labels_val.to(device)

            print('imname_train: {}'.format(imname_val))

            outputs = model(images_val)
            val_loss = loss_fn(input=outputs, target=labels_val)

            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels_val.data.cpu().numpy()
            
            dname = imname_val[0].split('.png')[0]
            np.save('{}/pred'.format(respath) + dname + '.npy', pred)
            np.save('{}/gt'.format(respath) + dname + '.npy', gt)
            np.save('{}/output'.format(respath) + dname + '.npy', outputs.data.cpu().numpy())

            running_metrics_val.update(gt, pred)
            val_loss_meter.update(val_loss.item())

    #writer.add_scalar('loss/val_loss', val_loss_meter.avg, i+1)
    #logger.info("Iter %d Loss: %.4f" % (i + 1, val_loss_meter.avg))
    print('Test loss: {}'.format(val_loss_meter.avg))

    score, class_iou = running_metrics_val.get_scores()
    for k, v in score.items():
        print('val_metrics, {}: {}'.format(k, v))

    for k, v in class_iou.items():
        print('val_metrics, {}: {}'.format(k, v))

    val_loss_meter.reset()
    running_metrics_val.reset()

#    if score["Mean IoU : \t"] >= best_iou:
#        best_iou = score["Mean IoU : \t"]
#        state = {
#            "epoch": i + 1,
#            "model_state": model.state_dict(),
#            "optimizer_state": optimizer.state_dict(),
#            "scheduler_state": scheduler.state_dict(),
#            "best_iou": best_iou,
#        }
#        save_path = os.path.join(writer.file_writer.get_logdir(),
#                                 "{}_{}_best_model.pkl".format(
#                                     cfg['model']['arch'],
#                                     cfg['data']['dataset']))
#        torch.save(state, save_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/pspnet_crop_ark.yml",
        help="Configuration file to use"
    )

    parser.add_argument('--areaname', nargs='?', type=str, default='ark', help='area name to specify the trained model.')

    args = parser.parse_args()

    print('areaname: {}'.format(args.areaname))

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    test(cfg, args.areaname)
