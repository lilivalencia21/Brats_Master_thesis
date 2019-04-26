from functions.nets import *
from functions.Training import *
from functions.patches import *

params_nnn = {'batch_size':128,
              'shuffle': True,
              'num_workers': 64}
#
experiment_nnn_cfg = {'patch_shape' : (32, 32, 32),
               'step' :  (12, 12, 12),
               'sampler' : UniformSampler,
               'epochs': 10,
               'model_name': 'CrossEntropyUnet3D',
                'patience': 3}

experiment_nnn_cfg.update({'sampler' : UniformSampler(experiment_nnn_cfg['patch_shape'], experiment_nnn_cfg['step'], num_elements=None)})