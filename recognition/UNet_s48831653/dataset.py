import numpy as np
import nibabel as nib
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import glob

###############################################################################
# 2D data loading #############################################################
###############################################################################

rangpur_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_"
slices_train_paths = glob.glob(rangpur_path + "train/*.nii.gz")
slices_validate_paths = glob.glob(rangpur_path + "validate/*.nii.gz")
slices_test_paths = glob.glob(rangpur_path + "test/*.nii.gz")
seg_train_paths = glob.glob(rangpur_path + "seg_train/*.nii.gz")
seg_validate_paths = glob.glob(rangpur_path + "seg_validate/*.nii.gz")
seg_test_paths = glob.glob(rangpur_path + "seg_test/*.nii.gz")

# Returns ndarray storing one-hot encoded layers for each segmentation category
def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    # retrieve the sorted unique elements of flattened arr
    # corresponds to the number of segmented categories
    channels = np.unique(arr)

    # res stores (height, width, channels)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)

    # store one-hot encoded layers for each category in res
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    return res


# Load 2D medical images
def load_data_2D(imageNames, normImage=False, categorical=False,
        dtype=np.float32, getAffines=False, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for
    each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive memory
    usage.

    imageNames: list of nifti file paths
    normImage: bool (normalise the image 0.0-1.0)
    categorical: if true, provided images are segmented and will be passed
    through to_channels
    getAffines: if true, also return a list of nifti affine matrices
    early_stop: Stop loading pre-maturely, leaving arrays mostly empry, for
    quick loading and testing scripts.
    '''
    affines = []
    num = len(imageNames)

    # determine image heigth x width (x channel) dimensions and construct
    # images darray to store
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0] #remove extra dimensions if present
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    # load each image and store in images darray
    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged') # read disk only
        affine = niftiImage.affine

        if len(inImage.shape) == 3:
            inImage = inImage[:,:,0] #remove extra dimensions if present
        inImage = inImage.astype(dtype)

        if normImage: # normalise
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)

        # store in images array
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i,:,:,:] = inImage
        else:
            images[i,:,:] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    return images

