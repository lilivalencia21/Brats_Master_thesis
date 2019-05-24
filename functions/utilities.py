import glob, os
import numpy as np
import nibabel as nib
import torch.nn as nn
import torch
import operator
import random
from functions.metrics_brats import *


def load_case(case_folder):
    case_dict = {}

    case_name = case_folder.split('/')[-1]

    case_dict['image_paths'] = [os.path.join(case_folder, case_name + "_flair.nii.gz"),
                                os.path.join(case_folder, case_name + "_t1.nii.gz"),
                                os.path.join(case_folder, case_name + "_t1ce.nii.gz"),
                                os.path.join(case_folder, case_name + "_t2.nii.gz")]

    case_dict['gt_path'] = [os.path.join(case_folder, case_name + "_seg.nii.gz")]

    case_dict['id'] = case_name

    data = [nib.load(images) for images in case_dict['image_paths']]

    case_dict['nifti_headers'] = list([img.header for img in data])

    case_dict['mean'] = np.stack([np.mean(img.get_data()) for img in data], axis=0).astype(np.float)

    case_dict['std_dev'] = np.stack([np.std(img.get_data()) for img in data], axis=0).astype(np.float)

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


def load_dataset(data_dir_train):
    dataset = []

    for case_name in os.listdir(data_dir_train):
        case_path = os.path.join(data_dir_train, case_name)

        dataset.append(load_case(case_path))

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
