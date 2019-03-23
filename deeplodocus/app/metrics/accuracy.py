import torch


def accuracy(output, labels):
    """
    AUTHORS:
    --------

    :author: Samuel Westlake
    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Compare the accuracy of the output tensor with the expected label tensor
    Work with any size of tensor
     - For images : Compute the pixel-wise label prediction accuracy
     - For single class : Compute the class accuracy

    :param output (tensor): Output of the model. Must be of shape (Batch size x num_classes x A) (A being any size of any dimension, e.g. A = h x w for an image)
    :param labels (tensor): Expected output of the model. Must be of shape (Batch size x A) or (Batch_size x 1 x A) (See above for A dim and size)

    RETURN:
    -------

    :return (tensor): The accuracy of the whole batch
    """
    _, output = output.max(1)
    return torch.mean((output == labels).float())
