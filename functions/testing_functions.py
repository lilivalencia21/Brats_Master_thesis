import torch.nn as nn
import torch
from functions.utilities import *


def segment_img(image_case, testing_cfg):

    model = testing_cfg['model']
    model.load_state_dict(torch.load(testing_cfg['model_path']))
    model.to(testing_cfg['device'])
    model.eval()
    device = testing_cfg['device']

    intensity_images = load_images(image_case['image_paths'])
    mean = test_case['mean']
    std_dev = test_case['std_dev']
    input_images = norm_array(intensity_images, mean, std_dev)

    img = np.expand_dims(input_images, axis=0)
    test_input = torch.tensor(img, dtype=torch.float32, requires_grad=False, device=device)

    with torch.no_grad():
        testing = model(test_input)

    testing_np = testing.cpu().detach().numpy()

    if testing_np[testing_np == 3]:
        testing_np[testing_np == 3] = 4

    results = np.argmax(testing_np, axis=1)

    segmentation_result = np.squeeze(results, axis=0)

    gt = load_images(image_case['gt_path'])
    dice = dice_multiclass(gt, segmentation_result)
    print('Dice for case {} is {}'.format(image_case['id'], dice))
    dices_file.write('{} \n {} \n'.format(image_case['id'], str(dice)))
    segm_name = '{}_seg.nii.gz'.format(image_case['id'])
    print('Saving image segmentation result as {}'.format(segm_name))
    save_segmentation_img(segmentation_result, nifti_image, testing_cfg['path_to_save_segm'],segm_name)

    print('Saving dice scores..........')
    dices_file.close()


