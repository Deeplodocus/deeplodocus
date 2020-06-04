import numpy as np
from typing import Tuple


def one_hot_encode(class_ids: int, num_classes: int) -> Tuple[np.array, None]:
    """
    AUTHORS:
    --------

    :author: Samuel Westlake
    :author: Alix Leroy

    DESCRIPTION:
    ------------

    One Hot Encoding has a binary output which sits into an orthogonal space.

    PARAMETERS:
    -----------

    :param class_ids (int): Label of the class
    :param num_classes(int): Number of classes

    RETURN:
    -------

    :return (np.array): The one hot encoded label
    :return: None
    """
    return np.eye(num_classes)[class_ids].astype(np.int64), None

