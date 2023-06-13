from setseed import set_seed

import torch

def dice_coef(y_true, y_pred, RANDOM_SEED = 21):
    set_seed(RANDOM_SEED)
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 1e-6
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)