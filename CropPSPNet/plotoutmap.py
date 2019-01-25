import numpy as np
from matplotlib import pyplot as plt
import os
import sys

matfpath = sys.argv[1]
dat = np.load(matfpath)

tagname = matfpath.split('/')[-1].replace('.npy', '') 

cmin = np.min(dat)
cmax = np.min(dat)
print('dat.shape: {}'.format(dat.shape))
print('dat.min: {}'.format(cmin))
print('dat.max: {}'.format(cmax))
for k in range(dat.shape[1]):
    pdat = dat[0, k, :, :]
    plt.imshow(pdat, cmap=plt.cm.jet, interpolation='nearest', aspect='equal')
    plt.axis('off')
    #plt.show()
    plt.savefig("plots/{}_{}.png".format(tagname, k+1))
    
