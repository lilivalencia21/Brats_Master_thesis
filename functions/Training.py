import time
from functions.utilities import cross_entropy_wrapper
import torch


def train_net(train_gen, val_gen, model, max_epochs, optimizer, device):
    print('Training started')
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

        #             # print statistics
        #             running_loss += loss.item()
        #             if i % 10 == 9:
        #                 end = time.time()
        #                 print('epoch={}'.format(epoch)+ '-'*10,
        #                           'loss[{}]'.format(i + 1), 'Training: {0:.5f}'.format(running_loss / 10))
        #                 running_loss = 0.0

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

        #                 loss_val = loss.item()

        #                 # print statistics
        #                 running_loss += loss.item()
        #                 if i % 10 == 9:
        #                     end = time.time()
        #                     print('epoch={}'.format(epoch)+ '-'*10,
        #                           'loss[{}]'.format(i + 1), 'Validation: {0:.5f}'.format(running_loss / 10))
        #                     running_loss = 0.0

        # print statistics
        print('epoch={}'.format(epoch) + '-' * 10,
              'loss', 'Validation: {0:.5f}'.format(running_loss / minibatches))
        current_loss = running_loss

        running_loss = 0.0

    print('Finished Training')