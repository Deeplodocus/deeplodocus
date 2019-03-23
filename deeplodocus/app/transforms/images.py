import os
import random
import numpy as np
import cv2
from typing import Union, List
from typing import Any
from typing import Tuple
from collections import OrderedDict

# Deeplodocus imports
from deeplodocus.utils.flags.lib import *

"""
This file contains all the default transforms for images
"""

def cloudy(images, clouds_dir):
    clouds = cv2.imread(random.choice(os.listdir(clouds_dir)))


def random_crop(image, output_size=None, output_ratio=None, scale=None):
    # Set the cropped image width and height (dx, dy)
    if output_size is not None:
        # If output_size is a list of lists, i.e. ((lower_bound, upper_bound), (lower_bound, upper_bound))
        if any(isinstance(item, list) or isinstance(item, tuple) for item in output_size):
            # Define a random output_size between the given bounds
            output_size = {
                np.random.randint(*output_size[0]),
                np.random.randint(*output_size[1])
            }
        # Calculate the height and width of the cropped patch
        dx = output_size[0]
        dy = output_size[1]
    elif output_ratio is not None:
        # If output_ratio is a list of lists, i.e. ((lower_bound, upper_bound), (lower_bound, upper_bound))
        if any(isinstance(item, list) or isinstance(item, tuple) for item in output_ratio):
            # Define a random output_ratio between the given bounds
            output_ratio = (
                image.shape[1] / np.random.randint(*output_size[0]),
                image.shape[0] / np.random.randint(*output_size[1])
            )
        # Calculate the height and width of the cropped patch
        dx = int(image.shape[1] / output_ratio[0])
        dy = int(image.shape[0] / output_ratio[1])
    elif scale is not None:
        if isinstance(scale, list) or isinstance(scale, tuple):
            # Define a random scale between the given bounds
            scale = np.random.randint(*scale)
        # Calculate the height and width of the cropped patch
        dx = int(image.shape[1] * scale)
        dy = int(image.shape[0] * scale)
    else:
        # Set the crop size to the size of the image
        dx, dy = image.shape[0:2]

    # Define the coordinates of the crop
    x0 = np.random.randint(0, image.shape[1] - dx)
    y0 = np.random.randint(0, image.shape[0] - dy)
    x1 = x0 + dx
    y1 = y0 + dy

    # Store the parameters that were selected
    transform = {
        "name": "crop",
        "method": crop,
        "module_path": __name__,
        "coords": (x0, y0, x1, y1)
        }

    # Return the cropped image and transform kwargs
    return crop(image, (x0, y0, x1, y1)), transform


def crop(image, coords):
    """
    :param image:
    :param coords:
    :return:
    """
    x0, y0, x1, y1 = coords
    return image[y0: y1, x0: x1, ...], None


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

    :param image: np.array: The image to transform
    :param kernel_size_min: int: Min size of the kernel
    :param kernel_size_max: int: Max size of the kernel


    RETURN:
    -------

    :return: The blurred image
    :return: The last transform data
    """
    kernel_size = (random.randint(kernel_size_min // 2, kernel_size_max // 2)) * 2 + 1
    image, _ = blur(image, kernel_size)
    transform = {"name": "blur",
                 "method": blur,
                 "module_path": __name__,
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

    :param image: np.array: The image to transform
    :param kernel_size: int: Kernel size

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

    :param image: np.array: The image to transform
    :param gamma: float: Gamma value

    RETURN:
    -------

    :return: The transformed image
    :return: None
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
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

    :param image: np.array: input image
    :param shape: tuple: target shape
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
    num_channels = 1 if image.ndim == 2 else image.shape[2]

    padded = np.zeros(shape)
    padded.fill(value)
    y0 = int((shape[0] - image.shape[0]) / 2)
    x0 = int((shape[1] - image.shape[1]) / 2)
    y1 = y0 + image.shape[0]
    x1 = x0 + image.shape[1]

    if num_channels == 1:
        padded[y0:y1, x0:x1, 0] = image
    else:
        padded[y0:y1, x0:x1, :] = image

    return padded, None


def channel_shift(image: np.array, shift: int)-> Tuple[np.array, None]:
    for ch in range(image.shape[2]):
        image[:, :, ch] += shift[ch]

    image[image < 0] = 0
    image[image > 255] = 255
    return image, None


def random_channel_shift(image: np.array, shift: int) ->Tuple[np.array, dict]:
    """
    AUTHORS:
    --------

    :author: Samuel Westlake
    :author: Alix Leroy

    DESCRIPTION:
    ------------

    :param image:
    :param shift:

    RETURN:
    -------

    :return image(np.array), transform(dict): The image with the channel shifted and the detail of the random transform
    """
    shift = np.random.randint(-shift, shift, image.shape[2])
    image, _ = channel_shift(image, shift)
    transform = {"name": "channel_shit",
                 "method": channel_shift,
                 "module_path": __name__,
                 "kwargs": {"shift": shift}}
    return image, transform


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
                 "module_path": __name__,
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
                 "module_path": __name__,
                 "kwargs": {"angle": angle}}
    return image, transform


def rotate(image: np.array, angle: float) -> Tuple[np.array, None]:
    """
    AUTHORS:

    :author: Samuel Westlake
    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Rotate an image

    PARAMETERS:
    -----------

    :param image (np.array): The input image
    :param angle (float): The rotation angle

    RETURN:
    -------

    :return: The rotated image
    :return: None
    """
    shape = image.shape
    rows, cols = shape[0:2]
    m = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(image, m, (cols, rows)), None


def normalize_image(
        image,
        mean: Union[None, list, int, float],
        standard_deviation: Union[float, int],
        cv_library: int = DEEP_LIB_OPENCV
) -> Tuple[Any, None]:
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
    :param mean: Union[None, list, int]: The mean of the channel(s)
    :param standard_deviation: int: The standard deviation of the channel(s)
    :param cv_library:

    RETURN:
    -------
    :return: a normalized image
    """


    if standard_deviation is None:
        #standard_deviation = 255
        standard_deviation = image.std(axis=(0, 1), keepdims=True)

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

        if isinstance(mean, list) or isinstance(mean, tuple):
            mean = mean[:channels]

    # Default option
    else:
        if mean is None:
            mean = image.mean(axis=(0, 1), keepdims=True) # Compute the mean on each channel


    normalized_image = (image - mean) / standard_deviation  # Norm = (data - mean) / standard deviation


    return normalized_image.astype(np.float32), None


def gaussian_blur(image: np.array, kernel_size: int) -> Tuple[np.array, None]:
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


def median_blur(image: np.array, kernel_size: int) -> Tuple[np.array, None]:

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


def bilateral_blur(image: np.array, diameter: int, sigma_color: int, sigma_space: int) -> Tuple[np.array, None]:
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Apply an bilateral blur to the image

    PARAMETERS:
    -----------

    :param image:
    :param diameter:
    :param sigma_color:
    :param sigma_space:

    RETURN:
    -------

    :return:
    """
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space), None


def grayscale(image: np.array) -> Tuple[np.array, None]:

    _, _, channels = image.shape

    if channels == 4:
        image, _ = convert_rgba2bgra(image)
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY), None
    elif channels == 3:
        image, _ = convert_rgba2bgra(image)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None
    else:
        return image, None


def convert_bgra2rgba(image):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Convert BGR(alpha) image to RGB(alpha) image

    PARAMETERS:
    -----------

    :param image: image to convert

    RETURN:
    -------

    :return: a RGB(alpha) image
    """

    # Get the number of channels in the image
    _, _, channels = image.shape

    # Handle BGR and BGR(A) images
    if channels == 3:
        image = image[:, :, (2, 1, 0)]
    elif channels == 4:
        image = image[:, :, (2, 1, 0, 3)]
    return image, None


def convert_rgba2bgra(image):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Convert RGB(alpha) image to BGR(alpha) image

    PARAMETERS:
    -----------

    :param image: image to convert

    RETURN:
    -------

    :return: a RGB(alpha) image
    """

    # Get the number of channels in the image
    _, _, channels = image.shape

    # Handle RGB and RGB(A) images
    if channels == 3:
        image = image[:, :, (2, 1, 0)]
    elif channels == 4:
        image = image[:, :, (2, 1, 0, 3)]
    return image, None


def remove_channel(image: np.array, index_channel: int):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Remove a channel from the image

    PARAMETERS:
    -----------

    :param image (np.array): The input image
    :param index_channel (int): The index of the channel to remove

    RETURN:
    -------

    :return matrix (np.array): The image without the selected channel
    """
    return np.delete(image, index_channel, -1), None

"""
"
" SEMANTIC SEGMENTATION
"
"""


def color2label(image: np.array, dict_labels: OrderedDict) -> np.array:
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Transform an RGB image to a array with the corresponding label indices

    PARAMETERS:
    -----------

    :param image (np.array): the input image
    :param indices (OrderedDict): An dictionary containing the relation class name => color (e.g. "cat" : [250, 89, 52]

    RETURN:
    -------

    :return labels (np.array): An array contianing the corresponding labels
    """
    # Convert float to integer
    image = image.astype(np.uint64)

    # Get image shape
    h, w, c = image.shape

    labels = np.zeros((h, w, 1), dtype=np.uint64)

    list_dict = list(dict_labels.values())

    # For each column and row
    for j in range(h):
        for i in range(w):

            # Get the current value
            v = image[j][i]

            # Get the corresponding label (need to convert the value to list other it doesn't work with ndarrays
            label = list_dict.index(list(v))

            # Add the corresponding label
            labels[j][i] = label

    return labels, None


def label2color(labels: np.array, dict_labels: OrderedDict) -> np.array:
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Transform an RGB image to a array with the corresponding label indices

    PARAMETERS:
    -----------

    :param labels (np.array): the array containing the labels
    :param indices (OrderedDict): An dictionary containing the relation class name => color (e.g. "cat" : [250, 89, 52]

    RETURN:
    -------

    :return image (np.array): The image with the colors corresponding to the given labels
    """
    # Convert float to integer
    labels = labels.astype(np.uint64)

    # Get image shape
    h, w, _ = labels.shape

    image = np.zeros((h, w, 3), dtype=np.uint8)

    list_dict = list(dict_labels.values())

    # For each column and row
    for j in range(h):
        for i in range(w):

            # Get the current value
            v = labels[j][i]

            # Get the corresponding color
            color = list_dict[int(v)]

            # Add the corresponding label
            image[j][i] = color

    return image, None