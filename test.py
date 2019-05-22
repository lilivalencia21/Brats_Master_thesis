import torch
import numpy as np
import nibabel as nib
from functions.utilities import *
from functions.instructions import *
from functions.patches import *
from functions.nets import UNet3D, UNet3DNNN
from functions.testing_functions import *
from functions.models_sergi import ResUnet, Unet3D
from functions.testing_functions import segment_img_patches

#To use with test_cross_validation function.
crossvalidation_cfg = {'model': UNet3D(),
                       'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                       'model_path': '/home/liliana/models/crossvalidation/validationData_from_5_to_9_fold_2.pt',
                       'training_set_txt':"/home/liliana/dataToValidate/cases_train_fold_2.txt",
                       'path_to_save_segm': "/home/liliana/Results/crossvalidation_img/Fold_2",
                       'path_to_save_txt': "/home/liliana/Results/crossvalidation_img/Fold_2/"+ 'fold_2.txt'}

crossval_optim_cfg = {'model': UNet3D(),
                       'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                       'model_path': "/home/liliana/models/crossvalidation_opt/validationData_from_0_to_4_fold_1.pt",
                       'training_set_txt':"/home/liliana/dataToValidate/optimizing/cases_train_fold_1.txt",
                       'path_to_save_segm': "/home/liliana/Results/crossval_optim/Fold_1/",
                       'path_to_save_txt': "/home/liliana/Results/crossval_optim/Fold_1/"+ 'fold_1.txt'}

testing_crossentropy = {'model': UNet3D(),
                       'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                       'model_path': "/home/liliana/models/CrossEntropyUnet3DModel/CrossEntropyUnet3D_from_15_to_19_fold_4.pt",
                       'training_set_txt':"/home/liliana/dataToValidate/CrossEntropyUnet3D_data/cases_train_fold_4.txt",
                       'path_to_save_segm':"/home/liliana/Results/CrossEntropyUnet3DResults/Fold4/",
                       'path_to_save_txt': "/home/liliana/Results/CrossEntropyUnet3DResults/Fold4/" + 'fold_4.txt'}

testing_diceLoss = {'model': UNet3D(),
                       'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                       'model_path': "/home/liliana/models/DiceLossUNet3D_Model/DiceLossUnet3D_from_0_to_4_fold_1.pt",
                       'training_set_txt':"/home/liliana/dataToValidate/DiceLossUNet3D_Data/cases_train_fold_1.txt",
                       'path_to_save_segm':"/home/liliana/Results/DiceLossUNet3DResults/Fold1/",
                       'path_to_save_txt': "/home/liliana/Results/DiceLossUNet3DResults/Fold1/" + 'fold_1.txt'}

testing_folder = {'model':UNet3D(),
                       'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                       'model_path': "/home/liliana/models/Unet100Cases_Model/UNet3N_from_0_to_24_fold_1.pt",
                       'training_set_txt':"/home/liliana/dataToValidate/Unet100Cases_Data/cases_train_fold_1.txt",
                       'path_to_save_segm':"/home/liliana/Results/Unet100casesResults/",
                       'path_to_save_metrics': "/home/liliana/Results/Unet100casesResults/"}


data_dir_test = "/home/liliana/Data/train"
dataset_test = load_dataset(data_dir_test)
# test_cross_validation(dataset_test, testing_folder)
cases_to_validate = "/home/liliana/dataToValidate/Unet100Cases_Data/cases_val_fold_1.txt"
with open(cases_to_validate) as f:
    validation_set = [line.rstrip('\n') for line in f]

dices_file = open(testing_folder['path_to_save_metrics'] + 'dice.txt', 'w')
hausdorff_file = open(testing_folder['path_to_save_metrics'] + 'hausdorff.txt', 'w')


for case_name in validation_set:
    case_data = get_by_id(dataset_test, case_name)
    dice, hd = segment_img(case_data, testing_folder)
    dices_file.write('{} \n {} \n'.format(case_name, str(dice)))
    hausdorff_file.write('{} \n {} \n'.format(case_name, str(hd)))

print('Saving metrics..........')
dices_file.close()
hausdorff_file.close()

