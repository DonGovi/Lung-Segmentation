#-*-coding:utf-8-*-

import SimpleITK as sitk
import pandas as pd
import numpy as np
import skimage
import os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_opening, convex_hull_image, disk
from skimage.morphology import binary_dilation as bd
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.misc
from . import CTViewer as cv


scan_file = "TIANCHI_data/train/LKDS-00001.mhd"

def load_scan(file):
    full_scan = sitk.ReadImage(scan_file)
    img_array = sitk.GetArrayFromImage(full_scan)  #numpy数组，z,y,x
    origin = np.array(full_scan.GetOrigin())[::-1]   #世界坐标原点 z,y,x
    old_spacing = np.array(full_scan.GetSpacing())[::-1]   #原体素间距
    return img_array, origin, old_spacing


def resample(image, old_spacing, new_spacing=[1, 1, 1]):
    '''
    将体素间距设为(1, 1, 1)
    '''
    resize_factor = old_spacing / new_spacing
    new_shape = image.shape * resize_factor
    new_shape = np.round(new_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = old_spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing
    

def plot_3d(image, threshold=-600):
    p = image.transpose(2,1,0)

    verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()


def largest_label_volume(image, bg=-1):
    vals, counts = np.unique(image, return_counts=True)    # 统计image中的值及频率
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -600, dtype=np.int8)+1
    # 获得阈值图像
    labels = measure.label(binary_image, connectivity=1)
    # label()函数标记连通区域
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0,0,0]
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image

def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1)>0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2)>2*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3,1)  
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10) 
    return dilatedMask

def extend_bounding(img):
    selem = disk(5)
    mask = bd(img, selem)

    return mask
    '''
    mask = np.copy(img)
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if(img[y,x]==1):
                if(img[y,x-1]==0 and x-10>=0):
                    mask[y,x-10:x]=1
                if(img[y,x+1]==0 and x+10<=img.shape[0]):
                    mask[y,x+1:x+11]=1
    return mask
    '''
def extend_mask(image):
    mask = np.copy(image)
    for i_layer in range(mask.shape[0]):
        slice_img = mask[i_layer]
        mask_img = extend_bounding(slice_img)
        mask[i_layer] = mask_img

    struct = generate_binary_structure(3,1)  
    dilatedMask = binary_dilation(mask,structure=struct,iterations=10) 
    return dilatedMask

if __name__ == '__main__':
    img_array, origin, old_spacing = load_scan(scan_file)
    image, new_spacing = resample(img_array, old_spacing)
    #segmented_lungs = segment_lung_mask(image, False)
    segmented_lungs_filled = segment_lung_mask(image, True)
    cv.view_CT(segmented_lungs_filled)

    process_lungs = extend_mask(segmented_lungs_filled)
    cv.view_CT(process_lungs)

#plot_3d(process_lungs, 0)