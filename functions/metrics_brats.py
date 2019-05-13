import numpy as np
from functions.medpy_hausdorff import hd as hausdorff_dist

def compute_multiclass_metrics(y_true, y_pred):
    """
        :param y_true: ground thruth segmentation [1,240, 240, 240]
        :param y_pred: Segmentation prediction after argmax and squeeze [1,240, 240, 240]
        :return: float, dice coefficient and hausdorff distance
        """
    y = y_pred.copy()
    nclasses = np.unique(y_true)
    y_true = y_true.squeeze(0)
    dice = []
    hausdorff_distance = []
    for c in nclasses:
        seg = y == c
        gt = y_true == c
        dice_class = dice_coef(gt, seg)
        dice.append(dice_class)
        hausdorff = hausdorff_dist(gt, seg)
        hausdorff_distance.append(hausdorff)

    return dice, hausdorff

def dice_coef(y_true, y_pred):
    intersection = np.sum(np.logical_and(y_true, y_pred))

    if intersection > 0:
        return (2.0 * intersection) / (np.sum(y_true>0) + np.sum(y_pred>0))
    else:
        return 0.0