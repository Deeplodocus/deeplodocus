import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from typing import Optional
from typing import List


class CrossEntropy2d(nn.Module):

    def __init__(self, weight: Optional[List] = None, size_average: bool = True, ignore_index: int = -100):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Apply a 2 dimensional CrossEntropy

        PARAMETERS:
        -----------

        :param weight (Optional[List]): A manual rescaling weights for each class
        :param size_average (bool): Divide the result by the number of pixel per image and per batch
        :param ignore_index (int): Index to ignore in the process (e.g. for unlabelled data)

        RETURN:
        -------

        :return: The 2D cross entropy
        """
        super(CrossEntropy2d, self).__init__()

        # Convert the list to tensor if not None
        if weight is not None:
            weight = torch.from_numpy(np.asarray(weight))

        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Compute the 2D Cross Entropy using the class attributes

        PARAMETERS:
        -----------

        :param outputs (tensor): The output tensor of the model
        :param targets (tensor): The target tensor

        RETURN:
        -------

        :return: The 2D Cross entropy
        """
        return cross_entropy2d(outputs, targets, self.weight, self.size_average, self.ignore_index)


def cross_entropy2d(outputs, targets, weight=None, size_average=True, ignore_index=-100):
    """
    AUTHORS:
    --------

    :author: Alix Leroy
    :author: Samuel Westlake

    DESCRIPTION:
    ------------

    Function to compute the 2D Cross entropy
    Modify the size of the output if required

    PARAMETERS:
    -----------

    :param outputs (tensor): The output tensor of the model
    :param targets(tenosr): The target tensor
    :param weight (tensor): A manual rescaling weights for each class
    :param size_average (bool): Average the resulting loss by the number of pixels in the images in the batch
    :param ignore_index (int): A possible index to ignore

    RETURN:
    -------

    :return loss(tensor): The 2D Cross Entropy
    """
    # Get the dimension of the outputs and the targets
    _, c, h, w = outputs.size()
    _, ht, wt = targets.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        outputs = F.interpolate(outputs, size=(ht, wt), mode="nearest", align_corners=True) # Upsample using only nearest in order not to have any unwanted interpolation between pixels

    # Flatten the outputs and targets
    outputs = outputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    targets = targets.view(-1)

    # Compute the 1D loss on the flatten tensors
    loss = F.cross_entropy(
        outputs, targets, weight=weight, size_average=size_average, ignore_index=ignore_index
    )
    return loss
