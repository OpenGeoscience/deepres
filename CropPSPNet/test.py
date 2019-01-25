import sys, os
import torch
import visdom
from osgeo import gdal

import argparse
import timeit
import numpy as np
import scipy.misc as misc
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.utils.data.dataloader import DataLoader as PytorchDataLoader
# Our libs
import json
from ptsemseg.myutils.config import Config
from ptsemseg.myutils.utils import update_config
from ptsemseg.tasks.transforms import augment_flips_color, augment_multiple_operations
from ptsemseg.dataset.image_provider import ImageProvider
from ptsemseg.dataset.threeband_image import ThreebandImageType
from ptsemseg.dataset.multiband_image import MultibandImageType
from ptsemseg.dataset.neural_dataset import TrainDataset, ValDataset
from matplotlib import pyplot as plt

from ptsemseg.metrics import runningScore, averageMeter

from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.utils import convert_state_dict

try:
    import pydensecrf.densecrf as dcrf
except:
    print(
        "Failed to import pydensecrf,\
           CRF post-processing will not work"
    )

def visualize_result(data, scoremaps, pred):
    #colors = loadmat('data/color150.mat')['colors']
    (img, imgname) = data

    print('imgname: {}'.format(imgname))

    probmaps = np.copy(scoremaps)
    theta = 0.75
    exp_a = np.exp(theta*probmaps[0, :, :])
    exp_b = np.exp(theta*probmaps[1, :, :])
    exp_c = np.exp(theta*probmaps[2, :, :])
    exp_sum = exp_a + exp_b + exp_c
    probmaps[0, :, :] = exp_a/exp_sum
    probmaps[1, :, :] = exp_b/exp_sum
    probmaps[2, :, :] = exp_c/exp_sum
    
    # prediction
    #pred_color = colorEncode(preds, colors)
    #---------------------------------------#
    #  Save ad Tif file.
    #---------------------------------------#
    tifname = imgname[0].replace('.png', '_PSPNet.tif')
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create('plots/'+tifname, probmaps.shape[2], probmaps.shape[1], 3,
            gdal.GDT_Float32)
    outband1 = outRaster.GetRasterBand(1)
    outband2 = outRaster.GetRasterBand(2)
    outband3 = outRaster.GetRasterBand(3)
    outband1.WriteArray(probmaps[0, :, :])
    outband2.WriteArray(probmaps[1, :, :])
    outband3.WriteArray(probmaps[2, :, :])
    outRaster.FlushCache()

    # aggregate images and save
    #im_vis = np.concatenate((img, pred_color),
    #                        axis=1).astype(np.uint8)
    #suffix = args.suffix.replace('.pth', '.png')
    imgname = imgname[0].replace('.png', '_pspnet.png') 
    print('img.name: {}'.format(imgname))
    print('probmaps.shape: {}'.format(probmaps.shape))
    fig = plt.figure(figsize=(20,20))
    plt.title(imgname)
    ax1 = plt.subplot(2,2,1)
    res=ax1.imshow(probmaps[0, :, :], cmap=plt.cm.jet, interpolation='nearest')
    ax1.set_title('background')
    cb = fig.colorbar(res)

    ax2 = plt.subplot(2,2,2)
    res=ax2.imshow(probmaps[1, :, :], cmap=plt.cm.jet, interpolation='nearest')
    ax2.set_title('building')
    cb = fig.colorbar(res)

    ax3 = plt.subplot(2,2,3)
    res=ax3.imshow(probmaps[2, :, :], cmap=plt.cm.jet, interpolation='nearest')
    ax3.set_title('bridge')
    cb = fig.colorbar(res)

    ax4 = plt.subplot(2,2,4)
    res=ax4.imshow(pred, cmap=plt.cm.jet, interpolation='nearest')
    ax4.set_title('prediction')
    cb = fig.colorbar(res)

    #fig.colorbar(probmaps[0, :, :], ax=ax1)
    #fig.colorbar(probmaps[1, :, :], ax=ax2)
    #fig.colorbar(probmaps[2, :, :], ax=ax3)
    #fig.colorbar(probmaps[3, :, :], ax=ax4)
        
    #plt.show()
    plt.savefig('plots/' + imgname)
    #img_name = info.split('/')[-1]
    #cv2.imwrite(os.path.join(args.result,
    #            img_name.replace('.jpg', '.png')), im_vis)

def index_in_copy_out_1024x1024(width):
    if width < 1024:
        print('Width is smaller than 1024. This is not enough to split')
        return None

    nwids = int(np.ceil(width/1024.0))
    cpwid = int(width/nwids)

    if cpwid%2 == 1:
        cpwid += 1
    
    insidx = [0]
    ineidx = [1024]
    cpsidx = [0]
    cpeidx = [cpwid]
    outsidx = [0]
    outeidx = [cpwid]

    for i in range(1, nwids-1, 1):
        ins = int((i+0.5)*cpwid - 512)
        ine = int(ins + 1024)
        cps = int(512-cpwid*0.5)
        cpe = int(512+cpwid*0.5)
        outs = int(cpwid*i)
        oute = int(cpwid*(i+1))

        insidx.append(ins)
        ineidx.append(ine)
        cpsidx.append(cps)
        cpeidx.append(cpe)
        outsidx.append(outs)
        outeidx.append(oute)

    rest = width - (nwids-1)*cpwid
    insidx.append(width-1024)
    ineidx.append(width)
    cpsidx.append(1024-rest)
    cpeidx.append(1024)
    outsidx.append(width-rest)
    outeidx.append(width)

    return insidx, ineidx, cpsidx, cpeidx, outsidx, outeidx


def index_in_copy_out(width):
    if width < 1024 :
        print('Width is smaller than 1024. This is not enough to split')
        return None

    kkwid = 1024
    nwids = int(np.ceil(width*1.0/kkwid))+3
    cpwid = int(width/nwids)

    if cpwid%2 == 1:
        cpwid += 1
    
    insidx = [0]
    ineidx = [kkwid]
    cpsidx = [0]
    cpeidx = [cpwid]
    outsidx = [0]
    outeidx = [cpwid]

    for i in range(1, nwids-1, 1):
        ins = int((i+0.5)*cpwid - kkwid/2)
        ine = int(ins + kkwid)
        cps = int(kkwid/2 - cpwid/2)
        cpe = int(kkwid/2 + cpwid/2)
        outs = int(cpwid*i)
        oute = int(cpwid*(i+1))

        insidx.append(ins)
        ineidx.append(ine)
        cpsidx.append(cps)
        cpeidx.append(cpe)
        outsidx.append(outs)
        outeidx.append(oute)

    rest = width - (nwids-1)*cpwid
    insidx.append(width-kkwid)
    ineidx.append(width)
    cpsidx.append(kkwid-rest)
    cpeidx.append(kkwid)
    outsidx.append(width-rest)
    outeidx.append(width)

    return insidx, ineidx, cpsidx, cpeidx, outsidx, outeidx


def test(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[: model_file_name.find("_")]
    print('model_file_name: {}'.format(model_file_name))
    print('model_name: {}'.format(model_name))

    # Setup image
    #print("Read Input Image from : {}".format(args.img_path))
    #img = misc.imread(args.img_path)

    #data_loader = get_loader(args.dataset)
    #data_path = get_data_path(args.dataset)
    #loader = data_loader(data_path, is_transform=True, img_norm=args.img_norm)
    #n_classes = loader.n_classes
    # Dataset and Loader
    #list_test = [{'fpath_img': args.test_img}]
    #dataset_val = TestDataset(
    #    list_test, args, max_sample=args.num_val)
#    paths = {
#        'masks': '/home/local/KHQ/chengjiang.long/Projects/core3d/data/AOIS/4AOIs/pngdata/refine_gtl',
#        'images': '/home/local/KHQ/chengjiang.long/Projects/core3d/data/AOIS/4AOIs/pngdata/rgb',
#        'ndsms': '/home/local/KHQ/chengjiang.long/Projects/core3d/data/AOIS/4AOIs/pngdata/ndsm',
#        'ndvis': '/home/local/KHQ/chengjiang.long/Projects/core3d/data/AOIS/4AOIs/pngdata/ndvi',
#        }
    
    paths = {
        'masks': '/data/CORE3D/AOIS/4AOIs/test_tmp/tile_building_fill',
        'images': '/data/CORE3D/AOIS/4AOIs/test_tmp/tile_image',
        'ndsms': '/data/CORE3D/AOIS/4AOIs/test_tmp/tile_dsm',
        'ndvis': '/data/CORE3D/AOIS/4AOIs/test_tmp/NDVI',
        }
    
    num_classes = 3
    ntest = len(os.listdir(paths['images']))
    test_idx = [i for i in range(ntest)]
    testds = ImageProvider(MultibandImageType, paths, image_suffix='.png')
    
    config_path = 'lcj_denseunet_1x1080_retrain.json'
    with open(config_path, 'r') as f:
        cfg = json.load(f)
        train_data_path = '/data/CORE3D/AOIS/Dayton_20sqkm/pngdata'
        print('train_data_path: {}'.format(train_data_path))
        dataset_path, train_dir = os.path.split(train_data_path)
        print('dataset_path: {}'.format(dataset_path) + ',  train_dir: {}'.format(train_dir))
        cfg['dataset_path'] = dataset_path
    config = Config(**cfg)

    config = update_config(config, num_channels=5, nb_epoch=50)
    #dataset_train = TrainDataset(trainds, train_idx, config, transforms=augment_flips_color)
    dataset_test = ValDataset(testds, test_idx, config)

    loader_test = data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=True)

    # Setup Model
    mymodel_dict = dict()
    mymodel_dict['arch'] = model_name
    model = get_model(mymodel_dict, num_classes)
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    for i, batch_data in enumerate(loader_test):
        #if batch_data['img_name'][0] != 'D4.png':
        #    continue

        # process data
        # print('batch_data: {}'.format(batch_data))
        #batch_data = batch_data[0]
        print('batch_data[img_data].size: {}'.format(batch_data['img_data'].shape))
        segSize = (batch_data['img_data'].shape[2],
                   batch_data['img_data'].shape[3])

        img = batch_data['img_data']
        gt = batch_data['seg_label'].data.cpu().numpy()
        dsize = img.shape
        predicted = np.zeros((dsize[0], num_classes, dsize[2], dsize[3]))
        predicted = torch.autograd.variable(torch.FloatTensor(predicted))

        with torch.no_grad():
            xinsidx, xineidx, xcpsidx, xcpeidx, xoutsidx, xouteidx = index_in_copy_out(segSize[0])
            yinsidx, yineidx, ycpsidx, ycpeidx, youtsidx, youteidx = index_in_copy_out(segSize[1])
            
            imgdata = torch.autograd.variable(torch.FloatTensor(img))
            print('imgdata.size: {}'.format(imgdata.size()))
            for i in range(len(xinsidx)):
                for j in range(len(yinsidx)):
                    samples = torch.autograd.Variable(imgdata[:,:,xinsidx[i]:xineidx[i], yinsidx[j]:yineidx[j]], volatile=True).cuda()
                    samples_img = samples.to(device)
                    # forward pass
                    prediction = model(samples_img)
                    #print('samples.size:  {}'.format(samples_img.size()))
                    #print('prediction.size:  {}'.format(prediction.size()))


                    #prediction = tmp_prediction.data.cpu().numpy()
                    #print('prediction.size: {}'.format(prediction.shape))
                    predicted[:, :, xoutsidx[i]:xouteidx[i], youtsidx[j]:youteidx[j]] = prediction[:,:,xcpsidx[i]:xcpeidx[i], ycpsidx[j]:ycpeidx[j]]

            #print('predicted: {}'.format(predicted.size()))
        running_metrics_test = runningScore(num_classes)
        print('gt.shape: {}'.format(gt.shape))
        print('predicted.shape: {}'.format(predicted.shape))

        pred = predicted.data.max(1)[1].cpu().numpy()
        running_metrics_test.update(gt, pred)
        print('score: {}'.format(running_metrics_test.get_scores()))

        probmaps = np.squeeze(predicted.data.cpu().numpy(), axis=0)
        print('preds.shape: {}'.format(probmaps.shape))
        
        pred = np.squeeze(pred, axis=0)
        visualize_result(
            (batch_data['img_data'], batch_data['img_name']),
            probmaps, pred)



        if args.dcrf:
            unary = predicted
            unary = np.squeeze(unary, 0)
            unary = -np.log(unary)
            unary = unary.transpose(2, 1, 0)
            w, h, c = unary.shape
            unary = unary.transpose(2, 0, 1).reshape(num_classes, -1)
            unary = np.ascontiguousarray(unary)

            resized_img = img

            d = dcrf.DenseCRF2D(w, h, loader.n_classes)
            d.setUnaryEnergy(unary)
            d.addPairwiseBilateral(sxy=5, srgb=3, rgbim=resized_img, compat=1)

            q = d.inference(50)
            mask = np.argmax(q, axis=0).reshape(w, h).transpose(1, 0)
            decoded_crf = loader.decode_segmap(np.array(mask, dtype=np.uint8))
            dcrf_path = args.out_path[:-4] + "_drf.png"
            misc.imsave(dcrf_path, decoded_crf)
            print("Dense CRF Processed Mask Saved at: {}".format(dcrf_path))

        #decoded = loader.decode_segmap(pred)
        print("Classes found: ", np.unique(pred))
        #misc.imsave(args.out_path, decoded)
        #print("Segmentation Mask Saved at: {}".format(args.out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="fcn8s_pascal_1_26.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--dataset",
        nargs="?",
        type=str,
        default="pascal",
        help="Dataset to use ['pascal, camvid, ade20k etc']",
    )

    parser.add_argument(
        "--img_norm",
        dest="img_norm",
        action="store_true",
        help="Enable input image scales normalization [0, 1] \
                              | True by default",
    )
    parser.add_argument(
        "--no-img_norm",
        dest="img_norm",
        action="store_false",
        help="Disable input image scales normalization [0, 1] |\
                              True by default",
    )
    parser.set_defaults(img_norm=True)

    parser.add_argument(
        "--dcrf",
        dest="dcrf",
        action="store_true",
        help="Enable DenseCRF based post-processing | \
                              False by default",
    )
    parser.add_argument(
        "--no-dcrf",
        dest="dcrf",
        action="store_false",
        help="Disable DenseCRF based post-processing | \
                              False by default",
    )
    parser.set_defaults(dcrf=False)

    parser.add_argument(
        "--img_path", nargs="?", type=str, default=None, help="Path of the input image"
    )
    parser.add_argument(
        "--out_path",
        nargs="?",
        type=str,
        default=None,
        help="Path of the output segmap",
    )
    args = parser.parse_args()
    test(args)
