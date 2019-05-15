import torch.nn as nn
import torch
import os
import nibabel as nib
import numpy as np
from functions.utilities import *
import copy
import math
from functions.patches import get_centers_unif, get_patch_slices



def segment_img(image_case, testing_cfg):

    model = testing_cfg['model']
    model.load_state_dict(torch.load(testing_cfg['model_path']))
    model.to(testing_cfg['device'])
    model.eval()
    device = testing_cfg['device']

    nifti_image = nib.load(image_case['image_paths'][0])
    intensity_images = load_images(image_case['image_paths'])
    mean = image_case['mean']
    std_dev = image_case['std_dev']
    input_images = norm_array(intensity_images, mean, std_dev)

    #Add 4 slices to z axis
    # norm_images = norm_array(intensity_images, mean, std_dev)
    # input_images = []
    # for img in norm_images:
    #     expand_z_axis = np.zeros((240, 240, 160))
    #     expand_z_axis[:, :, :155] = img
    #     input_images.append(expand_z_axis)

    img = np.expand_dims(input_images, axis=0)
    test_input = torch.tensor(img, dtype=torch.float32, requires_grad=False, device=device)

    with torch.no_grad():
        testing = model(test_input)

    testing_np = testing.cpu().detach().numpy()
    results = np.argmax(testing_np, axis=1)

    if np.any(results == 3):
        results[results == 3] = 4

    segmentation_result = np.squeeze(results, axis=0)

    # gt = load_images(image_case['gt_path'])
    # dice, hausdorff = compute_multiclass_metrics(gt, segmentation_result)
    # print('Dice for case {} is {}'.format(image_case['id'], dice))
    # print('Hausdorff for case {} is {}'.format(image_case['id'], hausdorff))

    segm_name = '{}.nii.gz'.format(image_case['id'])
    print('Saving image segmentation result as {}'.format(segm_name))
    save_segmentation_img(segmentation_result, nifti_image, testing_cfg['path_to_save_segm'],segm_name)

    # return dice, hausdorff

def load_validation_cases(dataset, training_set_txt):

    training_set = []
    with open(training_set_txt) as f:
        training_set = [line.rstrip('\n') for line in f]

    validation_set = []
    for case in dataset:
        if case['id'] not in training_set:
            validation_set.append(case)

    return validation_set

def save_segmentation_img(img_segm, original_img, path_to_save, segmentation_name):
    result_img = nib.Nifti1Image(img_segm, original_img.affine, original_img.header)
    image_filepath = os.path.join(path_to_save, segmentation_name)
    print("Saving {}...".format(segmentation_name))
    nib.save(result_img, image_filepath )

    print('Segmented image saved')

def segment_img_patches(image_case, testing_cfg):
    model = testing_cfg['model']
    model.load_state_dict(torch.load(testing_cfg['model_path']))
    model.to(testing_cfg['device'])
    model.eval()
    device = testing_cfg['device']

    nifti_image = nib.load(image_case['image_paths'][0])
    intensity_images = load_images(image_case['image_paths'])
    mean = image_case['mean']
    std_dev = image_case['std_dev']
    norm_case = norm_array(intensity_images, mean, std_dev)

    vol_shape = norm_case.shape[-3:]
    patch_shape = (32,32,32)
    unif_step = (32,32,32)
    centers = get_centers_unif(vol_shape, patch_shape, unif_step)
    slices = get_patch_slices(centers, patch_shape, step=None)

    output = torch.zeros(norm_case.shape)

    for slice_patch in slices:
        input_case = norm_case[slice_patch]

        img = np.expand_dims(input_case, axis=0)
        test_input = torch.tensor(img, dtype=torch.float32, requires_grad=False, device=device)

        with torch.no_grad():
            testing = model(test_input)

        testing_dimRed = torch.squeeze(testing, dim=0)
        output[slice_patch] = testing_dimRed

    testing_np = output.cpu().detach().numpy()
    results = np.argmax(testing_np, axis=0)

    if np.any(results == 3):
        results[results == 3] = 4

    # segmentation_result = np.squeeze(results, axis=0)



    segm_name = '{}.nii.gz'.format(image_case['id'])
    print('Saving image segmentation result as {}'.format(segm_name))
    save_segmentation_img(results, nifti_image, testing_cfg['path_to_save_segm'], segm_name)