import glob, os
import numpy as np
import nibabel as nib
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
# import matplotlib.pyplot as plt
import cv2
import torch.optim as optim
from functions import *
from functions.nets import UNet3D


# Load images for train, test and label
data_dir_train = "/home/liliana/Data/train"
data_dir_test = "/home/liliana/Data/valid"

list_dir_train = os.listdir(data_dir_train)
list_dir_test = os.listdir(data_dir_test)

#Load the images
# flair_train = load_img(data_dir_train, list_dir_train,"_flair.nii.gz")
# t1_train = load_img(data_dir_train, list_dir_train,"_t1.nii.gz")
# t1ce_train = load_img(data_dir_train, list_dir_train,"_t1ce.nii.gz")
# t2_train = load_img(data_dir_train, list_dir_train,"_t2.nii.gz")


#Create a matrix of lists
trainset = np.stack([flair_train,list(t1_train),list(t1ce_train),list(t2_train)],axis=1)
print( trainset.shape)


seg = list(map(lambda patient: load_brats_seg(os.path.join(data_dir_train, patient, patient + "_seg.nii.gz")),
               list_dir_train))

# Pad the image to extract the patches
pad_width = [(8,8),(8,8),(50,51)]

flair_pad = padding(flair_train, pad_width)
t1_pad = padding(t1_train, pad_width)
t1ce_pad = padding(t1ce_train, pad_width)
t2_pad = padding(t2_train, pad_width)

trainset_pad = np.stack([list(flair_pad),list(t1_pad),list(t1ce_pad),list(t2_pad)],axis=1)

seg_pad = np.stack(list(padding(seg, pad_width)))
print(seg_pad.shape)

# normalize the images
trainset = norm(trainset_pad)
# testset = norm(testset)

# Pass the net to the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = UNet3D()
net.to(device)

# create the optimizer
optimizer = optim.Adadelta(net.parameters())

# index = [0, 32, 64, 96, 128, 160, 192, 224]
index = [0, 64, 128, 192]
step = 64   #Original value 32
epochs = 20

sample_gt = []
sample_output = []

for epoch in range(epochs):

    running_loss = 0.0

    for i, (data, label_img) in enumerate(zip(trainset, seg_pad)):
        flair_img = data[0]
        t1_img = data[1]
        t1ce_img = data[2]
        t2_img = data[3]

        for j, start in enumerate(index):
            stop = start + step
            patch_slice = (slice(start, stop), slice(start, stop), slice(start, stop))
            # Extract patches of each modaly

            flair_slice = flair_img[patch_slice]

            img_norm = lambda x: (255 * (x - np.min(x)) / (np.max(x) - np.min(x))).astype(np.uint8)
            #             cv2.imwrite('flair_img_{}.jpg'.format(j),img_norm(flair_slice[:,:,0]))

            t1_slice = t1_img[patch_slice]

            t1ce_slice = t1ce_img[patch_slice]

            t2_slice = t2_img[patch_slice]

            label_slice = label_img[patch_slice]

            # Define the input of the net
            input_img = np.stack([flair_slice, t1_slice, t1ce_slice, t2_slice], axis=0)
            input_img = np.expand_dims(input_img, axis=0)

            optimizer.zero_grad()
            # forward + backward + optimize
            input_img = torch.tensor(input_img, dtype=torch.float32, requires_grad=True).to(device)
            target = torch.tensor(label_slice, dtype=torch.uint8).to(device)
            output = net(input_img)
            loss = cross_entropy_wrapper(output, target)
            loss.backward()
            optimizer.step()

            if j == 2:
                sample_output.append(output)
                sample_gt.append(target)


l=12
print(sample_output[6][0, :, 0, 0, 0])

# with open("sample_output.txt", "w") as output:
#     output.write(str(sample_output))

# print(sample_output[l][0, 0, :9, :9, 15])
# print(sample_gt[l][:9, :9, 15])

print('Finish')




