import random
import copy
import cv2
import numpy as np
from matplotlib import pyplot as plt

from scipy.misc import imresize

from ptsemseg.tasks.transforms import ToTensor
from .image_provider import AbstractImageProvider
from .image_cropper import ImageCropper


class Dataset:
    """
    base class for pytorch datasets
    """
    def __init__(self, image_provider: AbstractImageProvider, image_indexes, config, stage='train', transforms=ToTensor()):
        self.cropper = ImageCropper(config.target_rows,
                                    config.target_cols,
                                    config.train_pad if stage=='train' else config.test_pad,
                                    use_crop=True if stage=='train' else False)
        self.image_provider = image_provider
        self.image_indexes = image_indexes if isinstance(image_indexes, list) else image_indexes.tolist()
        if stage != 'train' and len(self.image_indexes) % 2: #todo bugreport it
            self.image_indexes += [self.image_indexes[-1]]
        self.stage = stage
        self.keys = {'image', 'image_name'}
        self.config = config
        self.transforms = transforms
        if transforms is None:
            self.transforms = ToTensor()

    def __getitem__(self, item):
        raise NotImplementedError


class TrainDataset(Dataset):
    """
    dataset for train stage
    """
    def __init__(self, image_provider, image_indexes, config, segm_downsampling_rate, stage='train', transforms=ToTensor()):
        super(TrainDataset, self).__init__(image_provider, image_indexes, config, stage, transforms=transforms)
        self.keys.add('mask')
        self.segm_downsampling_rate = segm_downsampling_rate

    def __getitem__(self, idx):
        im_idx = self.image_indexes[idx % len(self.image_indexes)]

        item = self.image_provider[im_idx]
        sx, sy = self.cropper.random_crop_coords(item.image)
        if self.cropper.use_crop and self.image_provider.has_alpha:
            for i in range(10):
                alpha = self.cropper.crop_image(item.alpha, sx, sy)
                if np.mean(alpha) > 5:
                    break
                sx, sy = self.cropper.random_crop_coords(item.image)
            else:
                return self.__getitem__(random.randint(0, len(self.image_indexes)))

        im = self.cropper.crop_image(item.image, sx, sy)
        mask = self.cropper.crop_image(item.mask, sx, sy)
        #print('a --> im.size = {},  mask.size = {}'.format(im.shape, mask.shape))

        #print('im.shape: {}'.format(im.shape) + ' <- item.image.shape: {}'.format(item.image.shape)
        #        + ', (x, y) = {}'.format((sx, sy)))
        bcheck_input = False
        if bcheck_input == True:
            rgb = im[:,:,0:3]
            ndsm = im[:,:,3]
            ndvi = im[:,:,4]

            img_rgb = copy.deepcopy(rgb)
            img_ndsm = copy.deepcopy(rgb)
            img_ndvi = copy.deepcopy(rgb)
            
            img_ndsm[:,:,0] = ndsm
            img_ndsm[:,:,1] = ndsm
            img_ndsm[:,:,2] = ndsm

            img_ndvi[:,:,0] = ndvi
            img_ndvi[:,:,1] = ndvi
            img_ndvi[:,:,2] = ndvi

            img_rgb[mask==255,0] = img_rgb[mask==255,0]*0.8
            img_rgb[mask==255,1] = img_rgb[mask==255,1]*0.8 + 255*0.2
            img_rgb[mask==255,2] = img_rgb[mask==255,2]*0.8
            
            img_ndsm[mask==255,0] = img_ndsm[mask==255,0]*0.8
            img_ndsm[mask==255,1] = img_ndsm[mask==255,1]*0.8 + 255*0.2
            img_ndsm[mask==255,2] = img_ndsm[mask==255,2]*0.8
            
            img_ndvi[mask==255,0] = img_ndvi[mask==255,0]*0.8
            img_ndvi[mask==255,1] = img_ndvi[mask==255,1]*0.8 + 255*0.2
            img_ndvi[mask==255,2] = img_ndvi[mask==255,2]*0.8
            
            figname = 'crop_' + str(im_idx) + '_sxsy_' + str(sx) + '_' + str(sy)
            cv2.imwrite('checktraindata/' + figname + '_ck1_RGB.png', img_rgb)
            cv2.imwrite('checktraindata/' + figname + '_ck2_NDSM.png', img_ndsm)
            cv2.imwrite('checktraindata/' + figname + '_ck3_NDVI.png', img_ndvi)

        im, mask = self.transforms(im, mask)
        #print('mask: {}'.format(mask))
        #print('size: {}'.format(mask.shape))
        target = np.zeros((mask.shape[1], mask.shape[2])).astype(np.uint8)
        for k in range(mask.shape[0]):
            target[mask[k, :, :] == 1] = k

        segm = imresize(target, (mask.shape[1] // self.segm_downsampling_rate, \
                                 mask.shape[2] // self.segm_downsampling_rate), \
                        interp='nearest')
        
        mask = None
        del mask

        #print('b --> im.size = {},  mask.size = {}, image_name: {}'.format(im.shape, target.shape, item.fn))
        
        #return {'img_data': im, 'seg_label': target, 'image_name': item.fn}
        output = dict()
        output['img_data'] = im
        output['img_name'] = item.fn
        output['seg_label'] = segm.astype(np.int)

        return output

    def __len__(self):
        return len(self.image_indexes) * max(self.config.epoch_size, 1) # epoch size is len images

class ValDataset(Dataset):
    """
    dataset for train stage
    """
    def __init__(self, image_provider, image_indexes, config, segm_downsampling_rate, stage='train', transforms=ToTensor()):
        super(ValDataset, self).__init__(image_provider, image_indexes, config, stage, transforms=transforms)
        self.keys.add('mask')
        self.segm_downsampling_rate = segm_downsampling_rate

    def __getitem__(self, idx):
        im_idx = self.image_indexes[idx % len(self.image_indexes)]

        item = self.image_provider[im_idx]
#        sx, sy = self.cropper.random_crop_coords(item.image)
#        if self.cropper.use_crop and self.image_provider.has_alpha:
#            for i in range(10):
#                alpha = self.cropper.crop_image(item.alpha, sx, sy)
#                if np.mean(alpha) > 5:
#                    break
#                sx, sy = self.cropper.random_crop_coords(item.image)
#            else:
#                return self.__getitem__(random.randint(0, len(self.image_indexes)))
#        
        sx = sy = 12
        im = self.cropper.crop_image(item.image, sx, sy)
        mask = self.cropper.crop_image(item.mask, sx, sy)
        #print('a --> image.size = {},  mask.size = {}'.format(item.image.shape, item.mask.shape))
        #print('b --> crop_im.size = {},  crop_mask.size = {}'.format(im.shape, mask.shape))

        #print('Press any key to continue ...')
        #input()

        #print('im.shape: {}'.format(im.shape) + ' <- item.image.shape: {}'.format(item.image.shape)
        #        + ', (x, y) = {}'.format((sx, sy)))
        im, mask = self.transforms(im, mask)
        #print('mask: {}'.format(mask))
        #print('size: {}'.format(mask.shape))
        target = np.zeros((mask.shape[1], mask.shape[2])).astype(np.uint8)
        for k in range(mask.shape[0]):
            target[mask[k, :, :] == 1] = k

        segm = imresize(target, (mask.shape[1] // self.segm_downsampling_rate, \
                                 mask.shape[2] // self.segm_downsampling_rate), \
                        interp='nearest')
        
        mask = None
        del mask

        #print('b --> im.size = {},  mask.size = {}, image_name: {}'.format(im.shape, target.shape, item.fn))
        
        #return {'img_data': im, 'seg_label': target, 'image_name': item.fn}
        output = dict()
        output['img_data'] = im
        output['img_name'] = item.fn
        output['seg_label'] = segm.astype(np.int)

        return output

    def __len__(self):
        return len(self.image_indexes) * max(self.config.epoch_size, 1) # epoch size is len images

class SequentialDataset(Dataset):
    """
    dataset for inference
    """
    def __init__(self, image_provider, image_indexes, config, stage='test', transforms=ToTensor()):
        super(SequentialDataset, self).__init__(image_provider, image_indexes, config, stage, transforms=transforms)
        self.good_tiles = []
        self.init_good_tiles()
        self.keys.update({'sy', 'sx'})

    def init_good_tiles(self):
        self.good_tiles = []
        for im_idx in self.image_indexes:
            item = self.image_provider[im_idx]

            positions = self.cropper.cropper_positions(item.image)
            
            if self.image_provider.has_alpha:
                item = self.image_provider[im_idx]
                alpha_generator = self.cropper.sequential_crops(item.alpha)
                for idx, alpha in enumerate(alpha_generator):
                    if np.mean(alpha) > 5:
                        self.good_tiles.append((im_idx, *positions[idx]))
            else:
                for pos in positions:
                    self.good_tiles.append((im_idx, *pos))

    def __getitem__(self, idx):
        if idx >= self.__len__():
            return None
        im_idx, sx, sy = self.good_tiles[idx]
        item = self.image_provider[im_idx]

        im = self.cropper.crop_image(item.image, sx, sy)

        im = self.transforms(im)
        
        #return {'image': im, 'startx': sx, 'starty': sy, 'image_name': item.fn}
        
        output = dict()
        output['img_data'] = im
        output['seg_label'] = segm.astype(np.int)

        return output

    def __len__(self):
        return len(self.good_tiles)


#class ValDataset(SequentialDataset):
#    """
#    dataset for validation
#    """
#    def __init__(self, image_provider, image_indexes, config, stage='test', transforms=ToTensor()):
#        super(ValDataset, self).__init__(image_provider, image_indexes, config, stage, transforms=transforms)
#        self.keys.add('mask')
#        self.segm_downsampling_rate = 1
#
#    def __getitem__(self, idx):
#        im_idx, sy, sx = self.good_tiles[idx]
#        item = self.image_provider[im_idx]
#
#        #im = self.cropper.crop_image(item.image, sx, sy)
#        #mask = self.cropper.crop_image(item.mask, sx, sy)
#        im = item.image
#        mask = item.mask
#        # print('im.shape: {}'.format(im.shape) + ' <- item.image.shape: {}'.format(item.image.shape)
#        #        + ', (x, y) = {}'.format((sx, sy)))
#        # cv2.imshow('w', im[...,:3])
#        # cv2.imshow('m', mask)
#        # cv2.waitKey()
#        im, mask = self.transforms(im, mask)
#        target = np.zeros((mask.shape[1], mask.shape[2]))
#        for k in range(mask.shape[0]):
#            target[mask[k, :, :] == 1] = k
#        
#        segm = imresize(target, (mask.shape[1] // self.segm_downsampling_rate, \
#                                 mask.shape[2] // self.segm_downsampling_rate), \
#                        interp='nearest')
#        mask = None
#        del mask
#        #return {'image': im, 'mask': target, 'startx': sx, 'starty': sy, 'image_name': item.fn}
#        output = dict()
#        output['img_data'] = im
#        output['img_name'] = item.fn
#        output['seg_label'] = target.astype(np.int)
#
#        return output
