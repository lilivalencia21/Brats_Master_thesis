import torch
import torch.nn as nn

def to_categorical(target):
    """
    :param target: target image tensor [bs, 1, patch_size_x, patch_size_y, patch_size_z]
    :return: tensor [bs, num_labels, patch_size_x, patch_size_y, patch_size_z]
    """
    target_channels = []
    for n in range(4):
        # target_channels.append((target[:, : , ...] == n).float())
        target_channels.append((target == n).float())
    target_cat = torch.cat(target_channels, dim=1)

    return target_cat

def cross_entropy_wrapper(pred, GT):
    """
    :param pred: model output
    :param GT: target patches
    :return: crossentropy loss
    """
    labels = GT.squeeze(1).long()
    loss = nn.CrossEntropyLoss()
    return loss(torch.log(torch.clamp(pred, 1E-7, 1.0)), labels)

def dice_loss(output, target):
    target_tocat = to_categorical(target)   #convert tensor from [bs, 1,..] to [bs, 5, ...]
    reduction_dim = (2, 3, 4)
    den = torch.sum(output * target_tocat, dim=reduction_dim)
    num = torch.sum(output, dim=reduction_dim) + torch.sum(target_tocat, dim=reduction_dim)
    loss = - torch.mean((2.0 / 4.0) * torch.sum((den/num), dim=1))
    return loss
