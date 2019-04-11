import time

from torch.utils.data import Dataset
from functions.patches import *
import torch

from abc import ABC, abstractmethod

class Sampler(ABC):
    @abstractmethod
    def get_centers(self, case):
        pass

class UniformSampler(Sampler):
    def __init__(self, patch_shape, unif_step, num_elements=None):
        self.patch_shape = patch_shape
        self.unif_step = unif_step
        self.num_elements = num_elements

    def get_centers(self, case):
        vol_shape = case['nifti_headers'][0]['dim'][1:4]
        centers = get_centers_unif(vol_shape, self.patch_shape, self.unif_step)

        return centers[:self.num_elements] if self.num_elements is not None else centers


class BalancedSampler(Sampler):
    def __init__(self, patch_shape):
        self.patch_shape = patch_shape

    def get_centers(self, case):
        vol_shape = case['nifti_headers'][0]['dim'][1:4]

        # Get class 0 centers
        pass

        centers = []
        return centers # Must return a list, where each element is a list of 3 floats


def generate_instruction(dataset, sampler, patch_shape):

    instructions = []

    for case in dataset:
        vol_shape = case['nifti_headers'][0]['dim'][1:4]
        data_centers = sampler.get_centers(case)
        data_slices = get_patch_slices(data_centers, patch_shape, step=None)

        for d in data_slices:
            instructions.append({'id' :case['id'], 'data_slice' :d})

    return instructions


class BratsDatasetLoader(Dataset):  # Inheritance
    def __init__(self, dataset, instructions):
        """
        Constructor of myDataset.
        """
        self.dataset = dataset

        print("Preloading images...")
        for case_idx in range(len(self.dataset)):
            self.dataset[case_idx]['images'] = load_images(self.dataset[case_idx]['image_paths'])
            #Added to try to optimize time
            self.dataset[case_idx]['norm_images']=norm_array(self.dataset[case_idx]['images'],self.dataset[case_idx]['mean'],
                                                             self.dataset[case_idx]['std_dev'])

            self.dataset[case_idx]['gt'] = load_images(self.dataset[case_idx]['gt_path'])

        self.instructions = instructions

    def __len__(self):
        """
        Returns an integer with the number of available patches
        """
        return len(self.instructions)

    def __getitem__(self, index):
        """
        Receives the index of the requested patch
        Returns a tuple (x, y) with input and ground truth patches
        """

        instruction = self.instructions[index]
        X, y = extract_patch(instruction, self.dataset)

        y = torch.Tensor(y).byte()
        X = torch.Tensor(X).float()

        return X, y