import time

from numpy import inf

from functions.utilities import cross_entropy_wrapper
import torch


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


