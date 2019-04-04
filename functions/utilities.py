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


def load_images(paths):
    images = np.stack([nib.load(modality_path).get_data() for modality_path in paths])

    #     cv2.imwrite('T1_img.jpg',images[0,:,:,80])

    #     if label_mapping is not None:
    #         for l_in, l_out in label_mapping.items():
    #             images[images==l_in]=l_out

    return images


def get_by_name(dataset, name):
    for case in dataset:
        if case['id'] is name:
            return case

def cross_entropy_wrapper(pred, GT):
    labels = GT.squeeze(1).long()
    loss = nn.CrossEntropyLoss()
    return loss(torch.log(torch.clamp(pred, 1E-7, 1.0)), labels)

def norm_array(image, mean, std):
    image_out = np.copy(image)
    for mod_idx, (m, s) in enumerate(zip(mean, std)):
        image_out[mod_idx] = (image_out[mod_idx] - m) / s
    return image_out

def norm(image, mean, std):
    return (image - mean) / std

def dice_coef(y_true, y_pred):
    # intersection = np.sum((y_true == y_pred)*[np.logical_or(y_true, y_pred)])
    intersection = np.sum(np.logical_and(y_true, y_pred))

    if intersection > 0:
        return (2.0 * intersection) / (np.sum(y_true>0) + np.sum(y_pred>0))
    else:
        return 0.0

def dice_multiclass(y_true, y_pred):

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

def save_segmentation_img(img_segm, original_img, path_to_save, segmetation_name):
    result_img = nib.Nifti1Image(img_segm, original_img.affine, original_img.header)
    image_filepath = os.path.join(path_to_save, segmetation_name)
    print("Saving {}...".format(segmetation_name))
    nib.save(result_img, image_filepath )

    print('Segmented image saved')

# def check_valid_samples(val_set, validation_txt):


def test_cross_validation(dataset, crossvalidation_cfg):

    model = crossvalidation_cfg['model']
    model.load_state_dict(torch.load(crossvalidation_cfg['model_path']))
    model.to(crossvalidation_cfg['device'])
    model.eval()
    device = crossvalidation_cfg['device']

    validation_set = load_validation_cases(dataset, crossvalidation_cfg['training_set_txt'])
    # dices = np.zeros((len(validation_set), 5))
    dices = []
    dices_file = open(crossvalidation_cfg['path_to_save_txt'], 'ab')

    for test_case in validation_set:
        nifti_image = nib.load(test_case['image_paths'][0])
        intensity_images = load_images(test_case['image_paths']).astype(np.float)
        mean = test_case['mean']
        std_dev = test_case['std_dev']
        input_images = np.stack([norm(image, mean, std) for image, mean, std in zip(intensity_images, mean, std_dev)])

        img = np.expand_dims(input_images, axis=0)
        test_input = torch.tensor(img, dtype=torch.float32, requires_grad=False, device=device)

        with torch.no_grad():
            testing = model(test_input)

        testing_np = testing.cpu().detach().numpy()


        results = np.argmax(testing_np, axis=1)
        segmentation_result = np.squeeze(results, axis=0)

        gt = load_images(test_case['gt_path'])
        dice = dice_multiclass(gt, segmentation_result)
        print('Dice for case {} is {}'.format(test_case['id'], dice))
        np.savetxt(dices_file, dice, fmt='%10.5f')
        segm_name = '{}_seg.nii.gz'.format(test_case['id'])
        print('Saving image segmentation result as {}'.format(segm_name))
        save_segmentation_img(segmentation_result, nifti_image, crossvalidation_cfg['path_to_save_segm'],segm_name)

    # print('Saving dice scores..........')
    # path = crossvalidation_cfg['path_to_save_txt']
    # np.savetxt(path, dices, fmt='%10.5f')
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