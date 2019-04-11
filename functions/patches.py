import copy
import itertools
from functions.utilities import *
import cv2



def get_patch_slices(centers, patch_shape, step=None):
    #     assert len(patch_shape) == centers.shape[1] == 3

    ### Compute patch sides for slicing
    half_sizes = [[dim // 2, dim // 2] for dim in patch_shape]
    for i in range(len(half_sizes)):  # If even dimension, subtract 1 to account for assymetry
        if patch_shape[i] % 2 == 0: half_sizes[i][1] -= 1

    # Actually create slices
    patch_locations = []
    for count, center in enumerate(centers):
        patch_slice = (slice(None),  # slice(None) selects all modalities
                       slice(center[0] - half_sizes[0][0], center[0] + half_sizes[0][1] + 1, step),
                       slice(center[1] - half_sizes[1][0], center[1] + half_sizes[1][1] + 1, step),
                       slice(center[2] - half_sizes[2][0], center[2] + half_sizes[2][1] + 1, step))
        patch_locations.append(patch_slice)


    return patch_locations


def get_centers_unif(vol_shape, patch_shape, unif_step):
    """
    :param vol_shape: tuple of 3 elements
    :param patch_shape: tuple of 3 elements
    :param unif_step: step between patches
    :return: list of  3D coordinates where the center of the patch will be located
    """
    assert len(vol_shape) == len(patch_shape) == len(unif_step)

    dim_ranges = []
    for dim in range(len(vol_shape)):
        dim_ranges.append([(patch_shape[dim] // 2) + 1,  # Dim start
                            vol_shape[dim] - (patch_shape[dim] // 2)])

    idx_x = range(dim_ranges[0][0], dim_ranges[0][1], unif_step[0])
    idx_y = range(dim_ranges[1][0], dim_ranges[1][1], unif_step[1])
    idx_z = range(dim_ranges[2][0], dim_ranges[2][1], unif_step[2])

    centers = [[x, y, z] for x, y, z in itertools.product(idx_x, idx_y, idx_z)]

    return centers


# def extract_patch(instruction, dataset):
#     # get case_dict with 'id'
#     case_dict = get_by_name(dataset, instruction['id'])
#
#     # get volume
#     volume_images = case_dict['images'].astype(np.float)
#     volume_gt = case_dict['gt']
#
#     # apply slice to volume
#     X_patch = copy.deepcopy(volume_images[instruction['data_slice']])
#
#     X_patch = norm_array(X_patch, case_dict['mean'], case_dict['std_dev'])
#
#     y_patch = copy.deepcopy(volume_gt[instruction['data_slice']])
#
#     return X_patch, y_patch

#Testing changes in function to optimize time
def extract_patch(instruction, dataset):
    # get case_dict with 'id'
    case_dict = get_by_name(dataset, instruction['id'])

    # get volume
    volume_images = case_dict['norm_images']
    volume_gt = case_dict['gt']

    # apply slice to volume
    # X_patch = copy.deepcopy(volume_images[instruction['data_slice']])
    X_patch = volume_images[instruction['data_slice']]
    # X_patch = norm_array(X_patch, case_dict['mean'], case_dict['std_dev'])
    y_patch = volume_gt[instruction['data_slice']]
    # y_patch = copy.deepcopy(volume_gt[instruction['data_slice']])

    return X_patch, y_patch
