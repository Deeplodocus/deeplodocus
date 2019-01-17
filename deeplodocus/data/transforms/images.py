import random
import numpy as np
import cv2
from typing import Union, List
from typing import Any
from typing import Tuple

# Deeplodocus imports
from deeplodocus.utils.flags.lib import *

"""
This file contains all the default transforms for images
"""


def random_blur(image: np.array, kernel_size_min: int, kernel_size_max: int) -> Tuple[Any, dict]:
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Apply a random blur to the image


    PARAMETERS:
    -----------

    :param image->np.array: The image to transform
    :param kernel_size_min->int: Min size of the kernel
    :param kernel_size_max->int: Max size of the kernel


    RETURN:
    -------

    :return: The blurred image
    :return: The last transform data
    """
    kernel_size = (random.randint(kernel_size_min // 2, kernel_size_max // 2)) * 2 + 1
    image, _ = blur(image, kernel_size)
    transform = {"name": "blur",
                 "method": blur,
                 "kwargs": {"kernel_size": kernel_size}}
    return image, transform


def blur(image: np.array, kernel_size: int) -> Tuple[Any, None]:
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Apply a blur to the image

    PARAMETERS:
    -----------

    :param image->np.array: The image to transform
    :param kernel_size->int: Kernel size

    RETURN:
    -------

    :return: The blurred image
    :return: None
    """
    return cv2.blur(image, (int(kernel_size), int(kernel_size))), None


def adjust_gamma(image, gamma):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Modify the gamma of the image

    PARAMETERS:
    -----------

    :param image -> np.array: The image to transform
    :param gamma -> float: Gamma value

    RETURN:
    -------

    :return: The transformed image
    :return: None
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table), None


def resize(image: np.array, shape, keep_aspect: bool = False, padding: int = 0):
    """
    AUTHORS:
    --------

    :author: Samuel Westlake
    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Resize an image

    PARAMETERS:
    -----------

    :param image->np.array: input image
    :param shape->tuple: target shape
    :param keep_aspect->bool, whether or not the aspect ration should be kept
    :param padding: int, value for padding if keep_aspect is True

    RETURN:
    -------

    :return->np.array: Image of size shape
    """
    # If we want to reduce the image
    if image.shape[0] * image.shape[1] > shape[0] * shape[1]:
        interpolation = cv2.INTER_LINEAR_EXACT  # Use the Bilinear Interpolation

    # If we prefer to increase the size
    else:
        interpolation = cv2.INTER_CUBIC  # Use the Bicubic interpolation

    # If we want to keep the aspect
    if keep_aspect:
        scale = min(np.asarray(shape[0:2]) / np.asarray(image.shape[0:2]))
        new_size = np.array(image.shape[0:2]) * scale
        image = cv2.resize(image, (int(new_size[1]), int(new_size[0])), interpolation=interpolation)
        image, _ = pad(image, shape, padding)

    else:
        image = cv2.resize(image, (shape[0], shape[1]), interpolation=interpolation)

    if image.ndim < 3:
        image = image[:, :, np.newaxis]

    return image, None



def pad(image, shape, value=0):
    """
    AUTHORS:
    --------

    :author: Samuel Westlake
    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Pads an image to self.x_size with a given value with the image centred

    PARAMETERS;
    -----------

    :param: image: input image
    :param: value

    RETURN:
    -------

    :return: Padded image
    :return: None
    """
    padded = np.empty(shape, dtype=np.uint8)
    padded.fill(value)
    y0 = int((shape[0] - image.shape[0]) / 2)
    x0 = int((shape[1] - image.shape[1]) / 2)
    y1 = y0 + image.shape[0]
    x1 = x0 + image.shape[1]

    nb_channels = padded.shape[2]

    if nb_channels == 1:
     padded[y0:y1, x0:x1, 0] = image
    else:

     padded[y0:y1, x0:x1, :] = image

    return padded.astype(np.float32), None


# TODO: Make the following method functional with Deeplodocus

# def random_channel_shift(image, shift):
#     shift = np.random.randint(-shift, shift, image.shape[2])
#     for ch in range(image.shape[2]):
#         image[:, :, ch] += shift[ch]
#     image[image < 0] = 0
#     image[image > 255] = 255
#     return image.astype(np.float32)


def random_rotate(image: np.array):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Rotate an image randomly

    PARAMETERS:
    -----------

    :param image: The image
    :param angle: The given angle

    RETURN:
    -------

    :return image->np.array: The rotated image
    :return transform->list: The info of the random transform
    """
    angle = (np.random.uniform(0.0, 360.0))
    image, _ = rotate(image, angle)
    transform = {"name": "rotate",
                 "method": rotate,
                 "kwargs": {"angle": angle}}
    return image, transform


def semi_random_rotate(image: np.array, angle: float):
    """
    AUTHORS:
    --------

    :author: Samuel Westlake
    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Rotate an image around a given angle

    PARAMETERS:
    -----------

    :param image: The image
    :param angle: The given angle

    RETURN:
    -------

    :return image->np.array: The rotated image
    :return transform->list: The info of the random transform
    """
    angle = (2 * np.random.rand() - 1) * angle
    image, _ = rotate(image, angle)
    transform = {"name": "rotate",
                 "method": rotate,
                 "kwargs": {"angle": angle}}
    return image, transform


def rotate(image: np.array, angle: float) -> Tuple[Any, None]:
    """
    AUTHORS:

    :author: Samuel Westlake
    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Rotate an image

    PARAMETERS:
    -----------

    :param image -> np.array: The input image
    :param angle -> float: The rotation angle

    RETURN:
    -------

    :return: The rotated image
    :return: None
    """
    shape = image.shape
    rows, cols = shape[0:2]
    m = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(image, m, (cols, rows)).astype(np.float32), None


def normalize_image(image, mean:Union[None, list, int], standard_deviation: float, cv_library: int = DEEP_LIB_OPENCV) ->Tuple[Any, None]:
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Normalize an image

    PARAMETERS:
    -----------

    :param image: an image
    :param mean->Union[None, list, int]: The mean of the channel(s)
    :param standard_deviation->int: The standard deviation of the channel(s)

    RETURN:
    -------
    :return: a normalized image
    """

    if standard_deviation is None:
        standard_deviation = 255

    # The normalization compute the mean of the image online if not given
    # The computed mean does not represent the mean of the dataset and therefore this is not recommended
    # This takes more time than just giving the mean as a parameter in the config file
    # However this time is still relatively small
    # Moreover this is done in parallel of the training
    # Note 1 : OpenCV is roughly 50% faster than numpy
    # Note 2 : Could be a limiting factor for big "mini"-batches (i.e. >= 1024) and big images (i.e. >= 512, 512, 3)

    # If OpenCV is selected (50% faster than numpy)
    if DEEP_LIB_OPENCV.corresponds(cv_library):
        channels = image.shape[-1]

        if mean is None:
            mean = cv2.mean(image)

        normalized_image = (image - mean[:channels]) / standard_deviation  # Norm = (data - mean) / standard deviation

    # Default option
    else:
        if mean is None:
            mean = np.mean(image, axis=(0, 1))  # Compute the mean on each channel

        normalized_image = (image - mean) / standard_deviation  # Norm = (data - mean) / standard deviation

    return normalized_image, None


def gaussian_blur(image : np.array, kernel_size : int) -> Tuple[Any, None]:
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Apply an gaussian blur to the image

    PARAMETERS:
    -----------

    :param image:
    :param kernel_size:

    RETURN:
    -------

    :return:
    """
    return cv2.GaussianBlur(image, (int(kernel_size), int(kernel_size)), 0), None


def median_blur(image : np.array, kernel_size : int) -> Tuple[Any, None]:
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Apply an medium blur to the image

    PARAMETERS:
    -----------

    :param image:
    :param kernel_size:

    RETURN:
    -------

    :return:
    """
    return cv2.medianBlur(image, int(kernel_size)), None
