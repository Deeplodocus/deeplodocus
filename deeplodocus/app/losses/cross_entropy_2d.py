import torch.nn.functional as F
import torch.nn as nn


class CrossEntropy2d(nn.Module):


    def __init__(self, weight=None, size_average=True):
        super(CrossEntropy2d).__init__()

        self.weight = weight
        self.size_average = size_average

    def forward(self, outputs, targets):

        return cross_entropy2d(outputs, targets, self.weight, self.size_average)


def cross_entropy2d(outputs, targets, weight=None, size_average=True):
    n, c, h, w = outputs.size()
    nt, ht, wt = targets.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        outputs = F.interpolate(outputs, size=(ht, wt), mode="bilinear", align_corners=True)

    outputs = outputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    targets = targets.view(-1)
    loss = F.cross_entropy(
        outputs, targets, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss