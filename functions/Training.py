import time

from numpy import inf
import numpy as np
from functions.utilities import cross_entropy_wrapper
import torch
import operator
import torch.optim as optim
from functions.instructions import *
from functions.nets import *


def train_net(train_gen, val_gen, model, max_epochs, optimizer, device):
    print('Training started')

    patience = 2
    tolerance = 1E-4
    last_loss = inf

    for epoch in range(max_epochs):

        running_loss = 0.0


        # Training
        minibatches = 0
        for minibatches, (local_batch, local_labels) in enumerate(train_gen):
            # print('minbatch:{}'.format(minibatches), sep='\r')

            local_batch, target = local_batch.to(device), local_labels.to(device)

            # forward + backward + optimize
            optimizer.zero_grad()

            output = model(local_batch)

            loss = cross_entropy_wrapper(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # print statistics
        print('epoch={}'.format(epoch) + '-' * 10,
              'loss', 'Training: {0:.5f}'.format(running_loss / minibatches))

        running_loss = 0.0
        minibatches = 0.0

        # Validation
        with torch.set_grad_enabled(False):
            for i, (local_batch, local_labels) in enumerate(val_gen):
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                # forward + backward + optimize
                output = model(local_batch)
                target = local_labels
                loss = cross_entropy_wrapper(output, target)

                minibatches = i
                running_loss += loss.item()

        # print statistics
        print('epoch={}'.format(epoch) + '-' * 10,
              'loss', 'Validation: {0:.5f}'.format(running_loss / minibatches))

        #Early stopping
        # if (abs(last_loss - running_loss) >= tolerance) and patience >= 2:
        #     break
        # else:
        #     last_loss = running_loss
        #     patience =+ 1

        running_loss = 0.0

    print('Finished Training')



def cross_validation(dataset, sampler, params, patch_shape, folds=4):

    indexes = np.linspace(start=0, stop=len(dataset) - 1, num=len(dataset), dtype='uint8')
    folds = np.arange(0, 4)
    num = int(len(indexes) / len(folds))

    # # Iterate over the dataset
    for i, fold in enumerate(folds):
        print('=====================================')
        print('Fold Number ', i + 1)
        print('=====================================')
        # Restart the model
        # model = UNet3D().to(device)

        # Get the indexes of the cases used for training. The rest are for validation
        val_idx = indexes[i * num:(i + 1) * num]
        # print('indexes of validation {}'.format(val_idx))
        train_idx = [index for index in indexes if index not in val_idx]
        # print('indexes of validation {}'.format(train_idx))

        # Loads training and validation data
        train_set = operator.itemgetter(*train_idx)(dataset)
        val_set = operator.itemgetter(*val_idx)(dataset)

        print("Generating training instructions...")
        instructions_train = generate_instruction(train_set, sampler, patch_shape)
        train_data = myDataset(train_set, instructions_train)
        train_gen = torch.utils.data.DataLoader(train_data, **params)
        print("Generated {} training instructions from {} images".format(len(instructions_train), len(train_set)))

        print("Generating validation instructions...")
        instructions_val = generate_instruction(val_set, sampler, patch_shape)
        val_data = myDataset(val_set, instructions_val)
        val_gen = torch.utils.data.DataLoader(val_data, **params)
        print("Generated {} validation instructions from {} images".format(len(instructions_val), len(val_set)))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = UNet3D()
        model.to(device)
        max_epochs = 10
        optimizer = optim.Adadelta(model.parameters())

        train_net(train_gen, val_gen, model, max_epochs, optimizer, device)

        # Save the model
        torch.save(model.state_dict(), '/home/liliana/models/net_norm_crossvalidation{}.pth'.format(i))



