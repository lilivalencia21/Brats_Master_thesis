import os
import numpy as np
import torch
from functions.utilities import load_dataset
from functions.instructions import BalancedSampler, UniformSampler
from functions.patches import *
from functions.nets import UNet3D
from functions.Training import cross_validation
from functions.loss_function import cross_entropy_wrapper, dice_loss
from functions.models_sergi import ResUnet, Unet3D



# data_dir_train = "/home/liliana/Data/train"
data_dir_train = "/home/liliana/Brats18TrainingData/"

print("Loading dataset...")
#load the dataset
dataset = load_dataset(data_dir_train)
print('Length of dataset is {}'.format(len(dataset)))

#Basic UNet parameters
params = {'batch_size':64,
              'shuffle': True,
              'num_workers': 64}
#
experiment_cfg = {'patch_shape' : (24, 24, 24),
               'step' :  (12, 12, 12),
               'sampler' : BalancedSampler,
               'model':UNet3D,
               'epochs': 10,
               'model_name': 'UNet50CasesLVR',
               'patience': 3,
               'pathToCasesNames':"/home/liliana/dataToValidate/Unet100Cases_Data/",
               'pathToSaveModel': "/home/liliana/models/Unet100Cases_Model/",
               'loss_function': dice_loss}

experiment_cfg.update({'sampler' : BalancedSampler(experiment_cfg['patch_shape'], 4, num_elements=2000)})


#Testing No New-net configuration

params_nnn = {'batch_size':2,
              'shuffle': True,
              'num_workers': 64}
#
experiment_nnn_cfg = {'patch_shape' : (128, 128, 128),
               'step' :  (16, 16, 16),
               'sampler' : UniformSampler,
               'model':UNet3D,
               'epochs': 15,
               'model_name': 'UNet3N',
               'patience': 5,
               'pathToCasesNames':"/home/liliana/dataToValidate/Unet100Cases_Data/",
               'pathToSaveModel': "/home/liliana/models/Unet100Cases_Model/",
               'loss_function': dice_loss}

# experiment_nnn_cfg.update({'sampler' : BalancedSampler(experiment_nnn_cfg['patch_shape'], 4, num_elements=1000)})
experiment_nnn_cfg.update({'sampler' : UniformSampler(experiment_nnn_cfg['patch_shape'], experiment_nnn_cfg['step'],
                                                      num_elements=None)})


cross_validation(dataset, params_nnn, experiment_nnn_cfg)