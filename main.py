import os
import numpy as np
from functions.utilities import *
from functions.instructions import *
from functions.patches import *
import torch
from functions.nets import *
from functions.Training import *
from functions.loss_function import *


data_dir_train = "/home/liliana/Data/train"
# data_dir_train = "/home/liliana/Brats18TrainingData/"

print("Loading dataset...")
#load the dataset
dataset = load_dataset(data_dir_train)
print('Length of dataset is {}'.format(len(dataset)))

# Parameters for dataloader
params = {'batch_size': 16,
          'shuffle': True,
          'num_workers': 64}

experiment_cfg = {'patch_shape' : (32, 32, 32),
               'step' : (12, 12, 12),
               'epochs': 10,
               'model_name': 'validationData',
                'patience': 3 }
experiment_cfg.update({'sampler' : UniformSampler(experiment_cfg['patch_shape'], experiment_cfg['step'], num_elements=None)})


# Experiment configuration No new-Net
params_nnn = {'batch_size':128,
              'shuffle': True,
              'num_workers': 64}
#
experiment_nnn_cfg = {'patch_shape' : (32, 32, 32),
               'step' :  (12, 12, 12),
               'sampler' : BalancedSampler,
               'epochs': 10,
               'model_name': 'testChanges',
               'patience': 3,
               'pathToCasesNames':"/home/liliana/dataToValidate/testfolder_Data/",
               'pathToSaveModel': "/home/liliana/models/testfolder_Model/",
               'loss_function': dice_loss}

experiment_nnn_cfg.update({'sampler' : BalancedSampler(experiment_nnn_cfg['patch_shape'], 4, num_elements=500)})


cross_validation(dataset, params_nnn, experiment_nnn_cfg)