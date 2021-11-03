import torch
import torch.nn as nn

import pdb

class WeightedL1Loss(nn.Module):
    def __init__(self, multiplier=5):
        super(WeightedL1Loss, self).__init__()
        self.criterion = torch.nn.L1Loss(reduction='none')
        self.multiplier = multiplier

    def forward(self, pred, gt, weight=None):
        loss = self.criterion(pred, gt)
        if weight is None:
            weight = gt
        loss = loss * torch.exp(self.multiplier * torch.abs(weight))
        loss = torch.mean(loss)
        return loss
