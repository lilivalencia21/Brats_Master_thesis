import glob, os
import numpy as np
import nibabel as nib
import torch.nn as nn
import torch
import operator
import random
from functions.metrics_brats import *
import itertools
import math



def load_case(case_folder, brain_mask=False):
    case_dict = {}

    case_name = case_folder.split('/')[-1]

    case_dict['image_paths'] = [os.path.join(case_folder, case_name + "_flair.nii.gz"),
                                os.path.join(case_folder, case_name + "_t1.nii.gz"),
                                os.path.join(case_folder, case_name + "_t1ce.nii.gz"),
                                os.path.join(case_folder, case_name + "_t2.nii.gz")]

    case_dict['gt_path'] = [os.path.join(case_folder, case_name + "_seg.nii.gz")]

    case_dict['id'] = case_name

    data_images = load_images(case_dict['image_paths'])
    data_gt = load_images(case_dict['gt_path'], GT=True)
    case_dict['images'] = data_images
    case_dict['gt'] = data_gt

    if brain_mask:
        brain_mask_slices = brain_box_img(data_images[0])
        case_dict['gt'] = np.stack([padd_img(img[brain_mask_slices]) for img in data_gt], axis=0)
        case_dict['images'] = np.stack([padd_img(img[brain_mask_slices]) for img in data_images], axis=0)


    # case_dict['nifti_headers'] = list([img.header for img in data])

    case_dict['mean'] = np.stack([np.mean(img) for img in case_dict['images']], axis=0)

    case_dict['std_dev'] = np.stack([np.std(img) for img in case_dict['images']], axis=0)

    case_dict['norm_images'] = norm_array(case_dict['images'], case_dict['mean'],  case_dict['std_dev'])

    # case_dict['norm_images'] = norm_array_feature_scaling(case_dict['images'])

    return case_dict


def split_dataset(dataset, train_percentage):
    n = len(dataset)
    idx = list(range(n))  # indices to all elements
    random.shuffle(idx)
    n_train = int(n * train_percentage)  # number train elements
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]
    train_set = operator.itemgetter(*train_idx)(dataset)
    val_set = operator.itemgetter(*val_idx)(dataset)

    return train_set, val_set


def load_dataset(data_dir_train, brain_mask=False):
    dataset = []
    bb = brain_mask
    for case_name in os.listdir(data_dir_train):
        case_path = os.path.join(data_dir_train, case_name)

        dataset.append(load_case(case_path, brain_mask=bb))

    return dataset


def load_images(paths, GT=False):

    images = np.stack([nib.load(modality_path).get_data() for modality_path in paths])

    if GT :
        images[images == 4] = 3

    return images


def get_by_id(dataset, name):
    for case in dataset:
        if case['id'] == name:
            return case

def norm_array(images, mean, std):
    """
    :param images: array of modality images
    :param mean: vector containing the mean of each modality
    :param std: vector containing the standard deviation of each modality
    :return: normalize images between 0 and 1
    """
    # image_out = np.copy(images)
    image_out = images.astype(np.float)
    for mod_idx, (m, s) in enumerate(zip(mean, std)):
        image_out[mod_idx] = (image_out[mod_idx] - m) / s
    return image_out

def norm_array_feature_scaling(images):
    """
    :param images: array of modality images
    :param mean: vector containing the mean of each modality
    :param std: vector containing the standard deviation of each modality
    :return: normalize images between 0 and 1
    """

    # image_out = np.copy(images)
    image_out = images.astype(np.float)
    for mod_idx in range(4):
        image_out[mod_idx] = (image_out[mod_idx] - image_out[mod_idx].min()) / \
                             (image_out[mod_idx].max()-image_out[mod_idx].min())
    return image_out


#Albert's function
def nic_binary_accuracy(y_pred, y_true, class_dim=1):
    """
    from Keras: K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))
    """
    y_true = torch.squeeze(y_true.long(), dim=class_dim)
    y_pred_categorical = torch.argmax(y_pred, dim=class_dim)
    return torch.mean(torch.eq(y_true, y_pred_categorical).float())

import sys


def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 25, fill = '='):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)

    bar = fill * filledLength + '>' * min(length - filledLength, 1) + '.' * (length - filledLength - 1)

    print('\r{} [{}] {}% {}'.format(prefix, bar, percent, suffix), end='\r')
    sys.stdout.flush()

    # Print New Line on Complete
    if iteration == total:
        print(' ')

def brain_box_img(imgage):
    bb = bbox2_ND(imgage)
    brain_box_vol_slice = (slice(bb[0], bb[1]), slice(bb[2], bb[3]), slice(bb[4], bb[5]))
    return brain_box_vol_slice


def bbox2_ND(img):
    N = img.ndim
    out = []
    for ax in itertools.combinations(reversed(range(N)), N - 1):
        nonzero = np.any(img, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])
    return tuple(out)

def padd_img(image):
    padd = np.pad(image,((16,16),(16,16),(16,16)), 'constant', constant_values=0)
    return padd

def remove_zeropad_volume(volume, patch_shape):
    # Get padding amount per each dimension
    selection = []
    for dim_size in patch_shape:
        slice_start = int(math.ceil(dim_size / 2.0))
        slice_stop = -slice_start if slice_start != 0 else None
        selection += [slice(slice_start, slice_stop)]
    volume = volume[tuple(selection)]
    return volume


