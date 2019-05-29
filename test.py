import torch
import numpy as np
import nibabel as nib
from functions.utilities import *
from functions.instructions import *
from functions.patches import *
from functions.nets import UNet3D
from functions.testing_functions import *
from functions.models_sergi import ResUnet, Unet3D
from functions.testing_functions import segment_img_patches

#To use with test_cross_validation function.

testing_folder = {'model':UNet3D(),
                       'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                       'model_path':"/home/liliana/models/FairResults28May_Model/Debbuging_from_142_to_212_fold_3.pt",
                       'path_to_save_segm':"/home/liliana/Results/FairResults28May/Fold3/",
                       'path_to_save_metrics':"/home/liliana/Results/FairResults28May/Fold3/"}


# data_dir_test = "/home/liliana/Data/train"
data_dir_test = "/home/liliana/Brats18TrainingData/"
dataset_test = load_dataset(data_dir_test)
# test_cross_validation(dataset_test, testing_folder)
cases_to_validate = "/home/liliana/dataToValidate/FairResults28May_Data/cases_val_fold_3.txt"
with open(cases_to_validate) as f:
    validation_set = [line.rstrip('\n') for line in f]

dices_file = open(testing_folder['path_to_save_metrics'] + 'dice.txt', 'w')
# hausdorff_file = open(testing_folder['path_to_save_metrics'] + 'hausdorff.txt', 'w')


for case_name in validation_set:
    case_data = get_by_id(dataset_test, case_name)
    dice = segment_img(case_data, testing_folder)
    dices_file.write('{} \n {} \n'.format(case_name, str(dice)))
    # hausdorff_file.write('{} \n {} \n'.format(case_name, str(hd)))

print('Saving metrics..........')
dices_file.close()
# hausdorff_file.close()

