# -*- coding: utf-8 -*-
import skimage.io
from skimage.future import graph
import skimage.color
import skimage.transform
import skimage.util
from skimage.segmentation import slic, mark_boundaries, felzenszwalb, find_boundaries
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import binary_dilation, disk


npy_file_path = "E:/lung_seg/"
lung_file = npy_file_path + "1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860.npy"
# Selective Search for lung CT

#def _generate_segments(img_orig, scale, sigma, min_si
    # open the Lung CT
def normalization(x):
    x = np.array(x, dtype=float)
    Min = np.min(x)
    Max = np.max(x)
    x = (x - Min) / (Max - Min)
    x = np.clip(x, 0, 1)
    return x

def slic_segment(lung_slice):
    labels = slic(lung_slice, n_segments=5000, sigma=1, multichannel=False, compactness=0.001, slic_zero=True,
              max_iter=20)
        
    seg_slice = mark_boundaries(lung_slice, labels, mode='subpixel')

    return seg_slice, labels

def find_neighbors(lung_slice, labels):
    vals = np.unique(labels, return_counts=False)    # count labels' values
    neighbors = np.zeros((len(vals), len(vals)))     # init neigborhood matrix

    selem = disk(2)
    for val in vals:
        temp = np.zeros(labels.shape)               # template matrix
        temp[labels==val] = 1                     # set all location 'val' to 1
        temp = binary_dilation(temp, selem)         # dilate the 'val' region
        extend_region = labels[temp==1]            # find the map of 'val' region after extended in labels
        neigborhood = np.unique(extend_region)    # find values in extend region
        print(val, neigborhood)
        for i in neigborhood:
            if(i > val):
            # i<val has been calculated before
                neighbors[val, i] = 1
                neighbors[i, val] = 1

    return neighbors




lung_array = np.load(lung_file)  
lung_slice = normalization(lung_array[:,:,80])
print("load lung file complete")
seg_slice, labels = slic_segment(lung_slice)
print("slic segmentation complete")

neighbors = find_neighbors(lung_slice, labels)
print(neighbors)


'''
fig,ax = plt.subplots(1,1,figsize=[10,10])
#ax.imshow(lung_slice ,cmap='gray')
#ax.imshow(labels, cmap='gray')
ax.imshow(seg_slice, cmap='gray')
#ax.imshow(boundaries, cmap='gray')

plt.show()
'''


