import glob, os
import numpy as np
import nibabel as nib
import torch.nn as nn
import torch
import operator
import random


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

    # case_dict['mean'] = [np.mean(img) for img in data].astype(np.float)
    #
    # case_dict['std_dev'] = [np.std(img) for img in data].astype(np.float)
    #
    # case_dict['norm'] = norm_array(data, case_dict['mean'],case_dict['std_dev'])

    case_dict['mean'] = np.stack(
        [np.mean(nib.load(modality_path).get_data()) for modality_path in case_dict['image_paths']], axis=0).astype(
        np.float)

    case_dict['std_dev'] = np.stack(
        [np.std(nib.load(modality_path).get_data()) for modality_path in case_dict['image_paths']], axis=0).astype(
        np.float)

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


def get_by_name(dataset, name):
    for case in dataset:
        if case['id'] is name:
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

def dice_coef(y_true, y_pred):
    intersection = np.sum(np.logical_and(y_true, y_pred))

    if intersection > 0:
        return (2.0 * intersection) / (np.sum(y_true>0) + np.sum(y_pred>0))
    else:
        return 0.0

def dice_multiclass(y_true, y_pred):
    """
    :param y_true: ground thruth segmentation [1,240, 240, 240]
    :param y_pred: Segmentation prediction after argmax and squeeze [1,240, 240, 240]
    :return: float, dice score
    """
    y = y_pred.copy()
    nclasses = np.unique(y_true)
    dice = []
    for c in nclasses:
        # y[y!=c] = 0
        seg = y == c
        gt = y_true == c
        dice_class = dice_coef(gt, seg)
        dice.append(dice_class)
    return dice

def save_segmentation_img(img_segm, original_img, path_to_save, segmentation_name):
    result_img = nib.Nifti1Image(img_segm, original_img.affine, original_img.header)
    image_filepath = os.path.join(path_to_save, segmentation_name)
    print("Saving {}...".format(segmentation_name))
    nib.save(result_img, image_filepath )

    print('Segmented image saved')


def test_cross_validation(dataset, crossvalidation_cfg):

    model = crossvalidation_cfg['model']
    model.load_state_dict(torch.load(crossvalidation_cfg['model_path']))
    model.to(crossvalidation_cfg['device'])
    model.eval()
    device = crossvalidation_cfg['device']

    validation_set = load_validation_cases(dataset, crossvalidation_cfg['training_set_txt'])
    dices_file = open(crossvalidation_cfg['path_to_save_txt'], 'w')

    for test_case in validation_set:
        nifti_image = nib.load(test_case['image_paths'][0])
        # intensity_images = load_images(test_case['image_paths']).astype(np.float)
        intensity_images = load_images(test_case['image_paths'])
        mean = test_case['mean']
        std_dev = test_case['std_dev']
        # input_images = np.stack([norm(image, mean, std) for image, mean, std in zip(intensity_images, mean, std_dev)])
        input_images = norm_array(intensity_images, mean, std_dev)

        img = np.expand_dims(input_images, axis=0)
        test_input = torch.tensor(img, dtype=torch.float32, requires_grad=False, device=device)

        with torch.no_grad():
            testing = model(test_input)

        testing_np = testing.cpu().detach().numpy()

        results = np.argmax(testing_np, axis=1)

        if np.any(results == 3):
            print('Detected 3')
            results[results == 3] = 4

        segmentation_result = np.squeeze(results, axis=0)

        gt = load_images(test_case['gt_path'])
        dice = dice_multiclass(gt, segmentation_result)
        print('Dice for case {} is {}'.format(test_case['id'], dice))
        dices_file.write('{} \n {} \n'.format(test_case['id'], str(dice)))
        segm_name = '{}_seg.nii.gz'.format(test_case['id'])
        print('Saving image segmentation result as {}'.format(segm_name))
        save_segmentation_img(segmentation_result, nifti_image, crossvalidation_cfg['path_to_save_segm'],segm_name)

    print('Saving dice scores..........')
    dices_file.close()

def load_validation_cases(dataset, training_set_txt):

    training_set = []
    with open(training_set_txt) as f:
        training_set = [line.rstrip('\n') for line in f]

    validation_set = []
    for case in dataset:
        if case['id'] not in training_set:
            validation_set.append(case)

    return validation_set

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
