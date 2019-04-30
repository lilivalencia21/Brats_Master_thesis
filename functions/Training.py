import time
from numpy import inf
import numpy as np
import torch
import operator
import torch.optim as optim
from functions.instructions import *
from functions.nets import *
from functions.loss_function import *
from functions.utilities import *


def train_net(train_gen, val_gen, model, max_epochs, optimizer, loss_function, device, model_name, model_path, patience=3):
    print('Training started')

    patience = patience

    # to track the training loss and accuracy as the model trains
    train_losses = []
    train_accuracies = []
    # to track the validation loss and accuracy as the model trains
    valid_losses = []
    valid_accuracies = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(model_name, model_path, patience=patience, verbose=True)

    for epoch in range(max_epochs):

        running_loss = 0.0
        running_accuracy = 0.0
        start = time.time()

        # Training
        minibatches = 0
        for minibatches, (local_batch, local_labels) in enumerate(train_gen):
            # print('minbatch:{}'.format(minibatches), sep='\r')

            local_batch, target = local_batch.to(device), local_labels.to(device)
            # forward + backward + optimize
            optimizer.zero_grad()
            output = model(local_batch)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            running_loss += loss.item()
            accuracy = nic_binary_accuracy(output, target)
            train_accuracies.append(accuracy)
            running_accuracy += accuracy.item()

            pref = 'Epoch={}'.format(epoch + 1)
            suff = 'loss --- ' + 'Training: {0:.5f}'.format(
                running_loss / (minibatches + 1)) + ' Accuracy: {0:.5f}'.format(running_accuracy / (minibatches + 1))
            printProgressBar(minibatches, len(train_gen), prefix=pref, suffix=suff)

        printProgressBar(len(train_gen), len(train_gen) )

        running_loss = 0.0
        minibatches = 0.0
        running_accuracy = 0.0


        # Validation
        with torch.set_grad_enabled(False):
            for i, (local_batch, local_labels) in enumerate(val_gen):
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                # forward + backward + optimize
                output = model(local_batch)
                target = local_labels
                loss = loss_function(output, target)
                minibatches = i
                valid_losses.append(loss.item())
                running_loss += loss.item()
                valid_loss = np.average(valid_losses)

                accuracy = nic_binary_accuracy(output, target)
                valid_accuracies.append(accuracy)
                running_accuracy += accuracy.item()

                suff = 'loss --- ' + 'Validation: {0:.5f}'.format(
                    running_loss / (minibatches + 1)) + ' Accuracy: {0:.5f}'.format(
                    running_accuracy / (minibatches + 1))
                pref = 'Epoch={}'.format(epoch+1)
                printProgressBar(minibatches, len(train_gen), prefix=pref, suffix=suff)

            printProgressBar(len(train_gen), len(train_gen))

        running_loss = 0.0
        running_accuracy = 0.0

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        train_accuracies = []
        valid_accuracies = []

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model, model_path)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        stop = time.time()
        print('Epoch time {0:.5f} seconds'.format((stop-start)))
        start = 0.0
        stop = 0.0

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(model_path + model_name))

    print('Finished Training')



def cross_validation(dataset, params, experiment_cfg, folds=4):

    start = time.time()

    indexes = np.linspace(start=0, stop=len(dataset) - 1, num=len(dataset), dtype='uint8')
    folds = np.arange(0, 4)
    num = int(len(indexes) / len(folds))

    # # Iterate over the dataset
    for i, fold in enumerate(folds):
        print('=====================================')
        print('Fold Number ', i + 1)
        print('=====================================')

        # Get the indexes of the cases used for training. The rest are for validation
        val_idx = indexes[i * num:(i + 1) * num]
        train_idx = [index for index in indexes if index not in val_idx]

        # Loads training and validation data
        train_set = operator.itemgetter(*train_idx)(dataset)
        val_set = operator.itemgetter(*val_idx)(dataset)

        train_cases = []
        for case in train_set:
            train_cases.append(case['id'])

        val_cases = []
        for case in val_set:
            val_cases.append(case['id'])

        # to change in future runnings to save in correct places
        trainStr = '\n'.join([str(elem) for elem in train_cases])
        file_train = open(experiment_cfg['pathToCasesNames']+'cases_train_fold_{}.txt'.format(i+1), 'w')
        file_train.write(trainStr)
        file_train.close()

        #to change in future runnings to save in correct places
        valStr = '\n'.join([str(elem) for elem in val_cases])
        file_val = open(experiment_cfg['pathToCasesNames']+ 'cases_val_fold_{}.txt'.format(i+1), 'w')
        file_val.write(valStr)
        file_val.close()

        sampler = experiment_cfg['sampler']

        print("Generating training instructions...")
        instructions_train = generate_instruction(train_set, sampler, experiment_cfg['patch_shape'])
        train_data = BratsDatasetLoader(train_set, instructions_train)
        train_gen = torch.utils.data.DataLoader(train_data, **params)
        print("Generated {} training instructions from {} images".format(len(instructions_train), len(train_set)))

        print("Generating validation instructions...")
        instructions_val = generate_instruction(val_set, sampler, experiment_cfg['patch_shape'])
        val_data = BratsDatasetLoader(val_set, instructions_val)
        val_gen = torch.utils.data.DataLoader(val_data, **params)
        print("Generated {} validation instructions from {} images".format(len(instructions_val), len(val_set)))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = UNet3D()
        # model = UNet3DNNN()
        model.to(device)

        max_epochs = experiment_cfg['epochs']
        optimizer = optim.Adadelta(model.parameters())
        loss_function = experiment_cfg['loss_function']
        model_name_fold = '{}_from_{}_to_{}_fold_{}.pt'.format(experiment_cfg['model_name'], val_idx[0], val_idx[-1], fold+1 )
        print ('The name of the model to save is: {}'.format(model_name_fold))
        model_path = experiment_cfg['pathToSaveModel']
        patience = experiment_cfg['patience']
        train_net(train_gen, val_gen, model, max_epochs, optimizer, loss_function, device, model_name_fold, model_path, patience)


    stop = time.time()
    print('total time for crossvalidation {0:.5f} minutes'.format((stop-start)/60))


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, checkpoint_model_name, model_path, patience=3, verbose=False ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.checkpoint_name = checkpoint_model_name

    def __call__(self, val_loss, model, model_path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_path)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_path)
            self.val_loss_min = val_loss
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model_path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), model_path + self.checkpoint_name)