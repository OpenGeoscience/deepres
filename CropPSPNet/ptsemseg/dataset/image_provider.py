import os
from .abstract_image_type import AbstractImageType
from typing import Type, Dict, AnyStr, Callable

class AbstractImageProvider:
    """
    base class for image providers
    """
    def __init__(self, image_type: Type[AbstractImageType], has_alpha=False):
        self.image_type = image_type
        self.has_alpha = has_alpha

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError



class ImageProvider(AbstractImageProvider):
    """
    provides images for dataset from disk
    """
    def __init__(self, image_type, paths, border=12, image_suffix=None, has_alpha=False):
        super(ImageProvider, self).__init__(image_type, has_alpha=has_alpha)
        self.im_names = os.listdir(paths['images'])
        if image_suffix is not None:
            self.im_names = [n.split('rgb')[1] for n in self.im_names if(image_suffix in n) and
            ('_01_' not in n) and ('_25_' not in n)]
        
        #print('im_names: {}'.format(self.im_names))
        #input()
        
        self.paths = paths
        self.border = border

    def __getitem__(self, item):
        return self.image_type(self.paths, self.im_names[item], self.border, self.has_alpha)

    def __len__(self):
        return len(self.im_names)
