import torch
import torch.nn as nn
import numpy as np

def to_categorical(target):
    """
    :param target: target image tensor [bs, 1, patch_size_x, patch_size_y, patch_size_z]
    :return: tensor [bs, num_labels, patch_size_x, patch_size_y, patch_size_z]
    """
    target_channels = []
    nclasses = np.unique(target.cpu().numpy())
    for n in range(4):
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

def dice_loss(output, target, smooth=0.0001):
    target_tocat = to_categorical(target)   #convert tensor from [bs, 1,..] to [bs, 5, ...]
    reduction_dim = (2, 3, 4)
    # nclasses = len(np.unique(target.cpu().numpy()))
    num = torch.sum(output * target_tocat, dim=reduction_dim)
    den = torch.sum(output, dim=reduction_dim) + torch.sum(target_tocat, dim=reduction_dim) + smooth

    loss = - 2.0 * (torch.mean(num/den, dim=1)).mean()

    num_dice_class = num
    den_dice_class = den

    with torch.set_grad_enabled(False):
        dice_class = 2.0 * torch.mean((num_dice_class/den_dice_class), dim=0).cpu().detach().numpy()

    return loss, dice_class

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def soft_dice(net_output, gt, smooth=1., smooth_in_nom=1.):
    axes = tuple(range(2, len(net_output.size())))
    target_tocat = to_categorical(gt)
    intersect = sum_tensor(net_output * target_tocat, axes, keepdim=False)
    denom = sum_tensor(net_output + target_tocat, axes, keepdim=False)
    result = (- ((2 * intersect + smooth_in_nom) / (denom + smooth))).mean()

    num_dice_class = intersect
    den_dice_class = denom

    with torch.set_grad_enabled(False):
        dice_class = torch.mean(((num_dice_class + smooth_in_nom) /(den_dice_class + smooth)), dim=0).cpu().detach().numpy()

    return result, dice_class


