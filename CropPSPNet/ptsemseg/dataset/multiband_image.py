import os
import numpy as np
import cv2
from .abstract_image_type import AbstractImageType


class MultibandImageType(AbstractImageType):
    """
    image type, that has dem/dtm information
    """
    def __init__(self, paths, fn, border, has_alpha):
        super().__init__(paths, fn, has_alpha)
        self.border = border
        self.img_data = None
        self.nir_data = None
        self.swir_data = None
        self.vh_data = None
        self.vv_data = None
        self.rede_data = None
        self.ndvi_data = None
        self.gtl_data = None

    def read_image(self):
        if self.img_data is None:
            self.img_data = cv2.imread(os.path.join(self.paths['images'], 'rgb' + self.fn), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if self.nir_data is None:
            self.nir_data = cv2.imread(os.path.join(self.paths['nirs'], 'nir' + self.fn), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if self.swir_data is None:
            dat1 = cv2.imread(os.path.join(self.paths['swirs'], 'swir1' + self.fn), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            dat2 = cv2.imread(os.path.join(self.paths['swirs'], 'swir2' + self.fn), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            self.swir_data = np.dstack([dat1, dat2])
        if self.vh_data is None:
            self.vh_data = cv2.imread(os.path.join(self.paths['vhs'], 'vh' + self.fn), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if self.vv_data is None:
            self.vv_data = cv2.imread(os.path.join(self.paths['vvs'], 'vv' + self.fn), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if self.rede_data is None:
            dat1 = cv2.imread(os.path.join(self.paths['redes'], 'rede1' + self.fn), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            dat2 = cv2.imread(os.path.join(self.paths['redes'], 'rede2' + self.fn), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            dat3 = cv2.imread(os.path.join(self.paths['redes'], 'rede3' + self.fn), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            self.rede_data = np.dstack([dat1, dat2, dat3])
        if self.ndvi_data is None:
            self.ndvi_data = cv2.imread(os.path.join(self.paths['ndvis'], 'ndvi' + self.fn), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        
        img_data = self.img_data
        nir_data = self.nir_data
        swir_data = self.swir_data
        vh_data = self.vh_data
        vv_data = self.vv_data
        rede_data = self.rede_data
        ndvi_data = self.ndvi_data
#        img_data = (np.float32(self.img_data.copy())/255.0 - 0.5)*2.0
#        nir_data = (np.float32(self.nir_data.copy())/255.0 - 0.5)*2.0
#        swir_data = (np.float32(self.swir_data.copy())/255.0 - 0.5)*2.0
#        rede_data = (np.float32(self.rede_data.copy())/255.0 - 0.5)*2.0
#        ndvi_data = (np.float32(self.ndvi_data.copy())/255.0 - 0.5)*2.0
        
#        print('img_data: {}'.format(img_data.shape))
#        print('nir_data: {}'.format(nir_data.shape))
#        print('swir_data: {}'.format(swir_data.shape))
#        print('rede_data: {}'.format(rede_data.shape))
#        print('ndvi_data: {}'.format(ndvi_data.shape))

        return self.finalyze(np.dstack([img_data, nir_data, swir_data, vh_data, vv_data, rede_data, ndvi_data]))

    def read_mask(self):
        elems = self.fn.split('.png')[0].split('_')
        gtimgname = 'gt_{}_{}.png'.format(elems[-2], elems[-1])
        if self.gtl_data is None:
            self.gtl_data = cv2.imread(os.path.join(self.paths['masks'], gtimgname), 0)
#            self.gtl_data = np.zeros(gtimg.shape)
#            mask1 = np.logical_and(gtimg[:,:,0]==255, gtimg[:,:,1]==255, gtimg[:,:,2]==0)
#            mask2 = np.logical_and(gtimg[:,:,0]==255, gtimg[:,:,1]==0, gtimg[:,:,2]==0)
#            mask3 = np.logical_and(gtimg[:,:,0]==0, gtimg[:,:,1]==255, gtimg[:,:,2]==0)
#            mask4 = np.logical_and(gtimg[:,:,0]==0, gtimg[:,:,1]==0, gtimg[:,:,2]==255)
#            self.gtl_data[mask1] = 1
#            self.gtl_data[mask2] = 2
#            self.gtl_data[mask3] = 3
#            self.gtl_data[mask4] = 4

        
        mask_bg = np.logical_or(self.gtl_data == 0, self.gtl_data == 2, self.gtl_data == 3).astype(np.uint8) * 255
        mask_corn = (self.gtl_data == 1).astype(np.uint8) * 255
        mask_soybean = (self.gtl_data == 2).astype(np.uint8) * 255
        
        mask = np.dstack([mask_bg, mask_corn, mask_soybean])
        return self.finalyze(mask)

    def read_alpha(self):
        if self.img_data is None:
            self.img_data = cv2.imread(os.path.join(self.paths['images'], self.fn), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        
        return self.finalyze(self.img_data[:,:,3])

    def finalyze(self, data):
        return self.reflect_border(data, b=self.border)

