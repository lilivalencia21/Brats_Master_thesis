import os
import numpy as np
from functions.utilities import *
from functions.instructions import *
from functions.patches import *
import torch
import torch.optim as optim
from functions.nets import *
from functions.Training import *


data_dir_train = "/home/liliana/Data/train"
# data_dir_train = "/home/liliana/Brats18TrainingData/"

print("Loading dataset...")
#load the dataset
dataset = load_dataset(data_dir_train)
print('Length of dataset is {}'.format(len(dataset)))
#Split the dataset in training and validation sets
train_percentage = 0.8
train_set, val_set = split_dataset(dataset, train_percentage)

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


#Experiment configuration No new-Net
params_nnn = {'batch_size':2,
          'shuffle': True,
          'num_workers': 64}
#
experiment_nnn_cfg = {'patch_shape' : (32, 32, 32),
               'step' :  (12, 12, 12),
               'sampler' : UniformSampler,
               'epochs': 10,
               'model_name': 'UNetDiceLoss',
                'patience': 3}

experiment_nnn_cfg.update({'sampler' : UniformSampler(experiment_nnn_cfg['patch_shape'], experiment_nnn_cfg['step'], num_elements=None)})


cross_validation(dataset, params, experiment_nnn_cfg)
# cross_validation(dataset, params_nnn, experiment_nnn_cfg)

# print("Generating training instructions...")
# instructions_train = generate_instruction(train_set, sampler, patch_shape)
# train_data = myDataset(train_set, instructions_train)
# train_gen = torch.utils.data.DataLoader(train_data, **params)
# print("Generated {} training instructions from {} images".format(len(instructions_train), len(train_set)))
#
# print("Generating validation instructions...")
# instructions_val = generate_instruction(val_set, sampler, patch_shape)
# val_data = myDataset(val_set, instructions_val)
# val_gen = torch.utils.data.DataLoader(val_data, **params)
# print("Generated {} validation instructions from {} images".format(len(instructions_val), len(val_set)))
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = UNet3D()
# model.to(device)
# max_epochs = 10
# optimizer = optim.Adadelta(model.parameters())

#
# print("Training...")
# Train_network = train_net(train_gen, val_gen, model, max_epochs, optimizer, device)
#
# #Save the model
# torch.save(model.state_dict(), "/home/liliana/models/net_norm_overlap.pth")




