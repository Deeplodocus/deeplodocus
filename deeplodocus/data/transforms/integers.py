import numpy as np


def one_hot_encode(labels, n_classes):
    """
    :param labels:
    :param n_classes:
    :return:
    """
    return np.eye(n_classes)[labels].astype(np.int64), None
