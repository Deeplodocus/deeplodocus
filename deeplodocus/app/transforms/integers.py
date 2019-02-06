import numpy as np


def one_hot_encode(class_ids, num_classes):
    """
    :param class_ids (int): Label of the class
    :param num_classes(int): Number of classes

    DESCRIPTION:
    ------------

    One Hot Encoding has a binary output which sits into an orthogonal space.

    :return:
    """
    return np.eye(num_classes)[class_ids].astype(np.int64), None

