import os
import numpy as np
import torch
from functions.utilities import load_dataset
from functions.instructions import BalancedSampler, UniformSampler
from functions.patches import *
from functions.nets import UNet3D
from functions.Training import cross_validation
from functions.loss_function import cross_entropy_wrapper, dice_loss, soft_dice
from functions.models_sergi import ResUnet, Unet3D




# data_dir_train = "/home/liliana/Data/train"
data_dir_train = "/home/liliana/Brats18TrainingData/"

print("Loading dataset...")
#load the dataset
dataset = load_dataset(data_dir_train, brain_mask=True)
print('Length of dataset is {}'.format(len(dataset)))


params_nnn = {'batch_size':128,
              'shuffle': True,
              'num_workers': 64}

experiment_nnn_cfg = {'patch_shape' : (32, 32, 32),
               'step' :  (16, 16, 16),
               'sampler' : UniformSampler,
               'model':UNet3D,
               'epochs': 10,
               'model_name': 'Debbuging',
               'patience': 5,
               'pathToCasesNames':"/home/liliana/dataToValidate/Debbuging_Data/",
               'pathToSaveModel': "/home/liliana/models/Debbuging_Model/",
               'loss_function': dice_loss}

# experiment_nnn_cfg = {'patch_shape' : (32, 32, 32),
#                'step' :  (16, 16, 16),
#                'sampler' : UniformSampler,
#                'model':UNet3D,
#                'epochs': 15,
#                'model_name': 'brain_mask',
#                'patience': 5,
#                'pathToCasesNames':"/home/liliana/dataToValidate/testfolder_Data/",
#                'pathToSaveModel': "/home/liliana/models/testfolder_Model/",
#                'loss_function': cross_entropy_wrapper}

# experiment_nnn_cfg.update({'sampler' : BalancedSampler(experiment_nnn_cfg['patch_shape'], 4, num_elements=1000)})
experiment_nnn_cfg.update({'sampler' : UniformSampler(experiment_nnn_cfg['patch_shape'], experiment_nnn_cfg['step'],
                                                      num_elements=None)})


cross_validation(dataset, params_nnn, experiment_nnn_cfg)