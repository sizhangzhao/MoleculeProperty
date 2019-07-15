import torch.nn.functional as F
import numpy as np


class PaddedL1Loss(object):

    def __init__(self):
        self.reduction = 'sum'

    def __call__(self, input, target, length):
        total_length = np.sum(np.array(length))
        return F.l1_loss(input, target, reduction=self.reduction) / total_length
