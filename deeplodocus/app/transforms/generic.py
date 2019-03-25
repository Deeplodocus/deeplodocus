import numpy as np
from typing import Tuple
from typing import Any
from typing import Union


def scale(item: Any, multiply: Union[float, int]=1, divide: Union[float, int]=1) -> Tuple[Any, None]:
    """
    AUTHORS:
    --------

    :author: Samuel Westlake

    DESCRIPTION:
    ------------

    Scale an item

    PARAMETERS:
    -----------

    :param item (Any):
    :param multiply (Union[float, int]):
    :param divide (Union[float, int]):

    RETURN:
    -------

    :return (Any): Scaled item
    :return: None
    """
    return item * multiply / divide, None


def bias(item, plus=0, minus=0):
    return item + plus - minus, None


def string2array(item, delimiter=",", cols=4, rows=50):
    output = np.zeros((rows, cols), dtype=np.float32)
    item = tuple(map(float, item.split(delimiter)))
    r = int(len(item) / cols)
    item = np.array(item).reshape(r, cols)
    output[0:r, :] = item
    return output, None


def reshape(item, shape):
    item = item.reshape(*shape)
    return item, None
