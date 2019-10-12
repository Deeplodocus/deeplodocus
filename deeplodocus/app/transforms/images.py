import random
import numpy as np
import cv2
from typing import Union, List
from typing import Any
from typing import Tuple
from collections import OrderedDict

# Deeplodocus imports
from deeplodocus.utils.notification import Notification
from deeplodocus.flags.notif import *
from deeplodocus.data.transform.transform_data import TransformData
from deeplodocus.flags.lib import *
"""
This file contains all the default transforms for images
"""


def random_crop(
        image: np.array,
        crop_size=None,
        crop_ratio=None,
        scale=None,
        resize=False
) -> Tuple[np.array, TransformData]:
    """
    AUTHORS:
    --------

    :author: Samuel Westlake

    DESCRIPTION:
    ------------

    Crop an image with random coordinates

    PARAMETERS:
    -----------

    :param image: (np.array): The input image to crop
    :param crop_size:
    :param crop_ratio:
    :param scale:
    :param resize: (bool) Whether or not to resize the image to the original shape after cropping

    RETURN:
    -------

    :return (np.array): The cropped image
    :return transform(dict): The parameters of the crop
    """
    # Set the cropped image width and height (dx, dy)
    if crop_size is not None:
        # If crop_size is a list of lists, i.e. ((lower_bound, upper_bound), (lower_bound, upper_bound))
        if any(isinstance(item, list) or isinstance(item, tuple) for item in crop_size):
            # Define a random crop_size between the given bounds
            crop_size = {
                np.random.randint(*crop_size[0]),
                np.random.randint(*crop_size[1])
            }
        # Calculate the height and width of the cropped patch
        dx = crop_size[0]
        dy = crop_size[1]
    elif crop_ratio is not None:
        # If crop_ratio is a list of lists, i.e. ((lower_bound, upper_bound), (lower_bound, upper_bound))
        if any(isinstance(item, list) or isinstance(item, tuple) for item in crop_ratio):
            # Define a random crop_ratio between the given bounds
            crop_ratio = (
                image.shape[1] / np.random.randint(*crop_size[0]),
                image.shape[0] / np.random.randint(*crop_size[1])
            )
        # Calculate the height and width of the cropped patch
        dx = int(image.shape[1] / crop_ratio[0])
        dy = int(image.shape[0] / crop_ratio[1])
    elif scale is not None:
        if scale > 1:
            Notification(DEEP_NOTIF_FATAL, " : random_crop : scale must be less than 1")
        if isinstance(scale, list) or isinstance(scale, tuple):
            # Define a random scale between the given bounds
            scale = np.random.random() * (scale[1] - scale[0]) + scale[0]
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
    transform = TransformData(name="crop",
                              method=crop,
                              module_path=__name__,
                              kwargs={
                                  "coords": (x0, y0, x1, y1),
                                  "resize": True
                                  }
                              )

    cropped_image, _ = crop(image, (x0, y0, x1, y1), resize=resize)

    # Return the cropped image and transform kwargs
    return cropped_image, transform


def crop(image: np.array, coords: Union[List, Tuple], resize: bool = False) -> Tuple[np.array, None]:
    """
    AUTHORS:
    --------

    :author: Samuel Westlake
    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Crop an image at specific coordinates

    PARAMETERS:
    -----------

    :param image:  (np.array): The input image
    :param coords: (list): The coordinates to be cropped
    :param resize: (bool) Whether or not to resize the image to the original shape after cropping

    RETURN:
    -------

    :return (np.array): The cropped image
    :return: None
    """
    x0, y0, x1, y1 = coords
    if resize:
        height, width = image.shape[0:2]
        return cv2.resize(image[y0: y1, x0: x1, ...], (width, height)), None
    else:
        return image[y0: y1, x0: x1, ...], None


def random_blur(image: np.array, kernel_size_min: int, kernel_size_max: int) -> Tuple[Any, TransformData]:
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
    # Compute kernel size
    kernel_size = (random.randint(kernel_size_min // 2, kernel_size_max // 2)) * 2 + 1

    # Blur the image
    image, _ = blur(image, kernel_size)

    # Store the parameters of the random blur
    transform = TransformData(name="blur",
                              method=blur,
                              module_path=__name__,
                              kwargs={
                                  "kernel_size": kernel_size
                                  }
                              )

    # Return the blurred image and the stored parameters
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

    :param image (np.array): The image to transform
    :param kernel_size (int): Kernel size

    RETURN:
    -------

    :return: The blurred image
    :return: None
    """
    return cv2.blur(image, (int(kernel_size), int(kernel_size))), None


def adjust_gamma(image, gamma) -> Tuple[np.array, None]:
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


def resize(image: np.array, shape, keep_aspect: bool = False, padding: int = 0, method=None) -> Tuple[np.array, None]:
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

    :return (np.array): Image of size shape
    :return: None
    """
    if method is None:
        # If we want to reduce the image
        if image.shape[0] * image.shape[1] > shape[0] * shape[1]:
            method = cv2.INTER_LINEAR_EXACT  # Use the Bilinear Interpolation
        # If we prefer to increase the size
        else:
            method = cv2.INTER_CUBIC  # Use the Bicubic interpolation

    # TODO : Get the method in a more pythonic way
    else:
        if method == "nearest":
            method = cv2.INTER_NEAREST
        elif method == "linear":
            method = cv2.INTER_LINEAR
        else:
            method = cv2.INTER_CUBIC

    # If we want to keep the aspect
    if keep_aspect:
        scale = min(np.asarray((shape[1], shape[0])) / np.asarray(image.shape[0:2]))
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=method)
        image, _ = pad(image, shape, padding)

    else:
        image = cv2.resize(image, (shape[0], shape[1]), interpolation=method)

    if image.ndim < 3:
        image = image[:, :, np.newaxis]

    return image, None


def pad(image, shape, value: int = 0) -> Tuple[np.array, None]:
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
    :param shape: (np.array): input image
    :param image: (np.array): input image
    :param value:  (int): Padding value

    RETURN:
    -------

    :return padded(np.array): Padded image
    :return: None
    """
    padded = np.zeros((int(shape[1]), int(shape[0])))
    padded.fill(value)
    y0 = int((shape[1] - image.shape[0]) / 2)
    x0 = int((shape[0] - image.shape[1]) / 2)
    y1 = y0 + image.shape[0]
    x1 = x0 + image.shape[1]
    padded[y0:y1, x0:x1, ...] = image
    return padded, None


def channel_shift(image: np.array, shift: int)-> Tuple[np.array, None]:
    """
    AUTHORS:
    --------

    :author: Samuel Westlake
    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Shift channels of a particular intensity value

    PARAMETERS:
    -----------

    :param image (np.array): The input image
    :param shift (int): The intensity of the shift

    RETURN:
    -------

    :return image(np.array): The image with the shifted channels
    :return: None
    """

    # Shift the channel of X intensity
    for ch in range(image.shape[2]):
        image[:, :, ch] += shift[ch]

    # Saturate channels too large
    image[image < 0] = 0
    image[image > 255] = 255
    return image, None


def random_channel_shift(image: np.array, shift: int) ->Tuple[np.array, TransformData]:
    """
    AUTHORS:
    --------

    :author: Samuel Westlake
    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Select a random value within a given range and shift all the channel of the image of this intensity value

    PARAMETERS:
    -----------

    :param image:
    :param shift (int): The index

    RETURN:
    -------

    :return image(np.array): The image with the channel shifted
    :return transform (TransformData): The parameters of the random shift
    """
    # Get the shifting value
    shift = np.random.randint(-shift, shift, image.shape[2])

    # Shift the channel
    image, _ = channel_shift(image, shift)

    # Store the parameters
    transform = TransformData(name="channel_shift",
                              method=channel_shift,
                              module_path=__name__,
                              kwargs={
                                  "shift": shift
                                  }
                              )

    # Return the image with the shift channel and the stored parameters
    return image, transform


def random_rotate(image: np.array) ->Tuple[np.array, TransformData]:
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

    :return image (np.array): The rotated image
    :return transform (TransformData): The info of the random transform
    """
    # Pick a random angle value
    angle = (np.random.uniform(0.0, 360.0))

    # Rotate the image
    image, _ = rotate(image, angle)

    # Store the random transform parameters
    transform = TransformData(name="rotate",
                              method=rotate,
                              module_path=__name__,
                              kwargs={
                                  "angle": angle
                                  }
                              )

    # return the rotated image and the random parameters
    return image, transform


def semi_random_rotate(image: np.array, angle: float) -> Tuple[np.array, TransformData]:
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

    :return image(np.array): The rotated image
    :return transform(TransformData): The info of the random transform
    """

    # Select a random angle within a given range
    angle = (2 * np.random.rand() - 1) * angle

    # Rotate the image
    image, _ = rotate(image, angle)

    # Store the random parameters
    transform = TransformData(name="rotate",
                              method=rotate,
                              module_path=__name__,
                              kwargs={
                                  "angle": angle
                                  }
                              )

    # Return the rotated image and the random parameters
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
    # Get the image shape
    shape = image.shape

    # Get the number of columns and rows
    rows, cols = shape[0:2]

    # Create an affine transformation
    m = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

    # Transform the image and return it
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
    :return normalized_image (np.array): a normalized image
    :return: None
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

    :param image (np.array): Input image
    :param kernel_size (int): Size of the kernel

    For more information, please check [OpenCV documentation on the median filter](    https://docs.opencv.org/4.0.1/d4/d86/group__imgproc__filter.html#ga564869aa33e58769b4469101aac458f9)


    RETURN:
    -------

    :return (np.array): The blurred image
    :return: None
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

    :param image (np.array): The input image
    :param diameter (int): Diameter of the kernel
    :param sigma_color (int):  Sigma value of the color space
    :param sigma_space (int):  Sigma value of the coordinate space

    For more information, please check [OpenCV documentation on the bilateral filter](https://docs.opencv.org/4.0.1/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed)

    RETURN:
    -------

    :return (np.array): The blurred image
    :return: None
    """
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space), None


def grayscale(image: np.array) -> Tuple[np.array, None]:
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Convert an RGB(a) image to grayscale

    PARAMETERS:
    -----------

    :param image (np.array): The image to transform to grayscale

    RETURN:
    -------

    :return image(np.array): The grayscale image
    :return: None
    """

    _, _, channels = image.shape

    if channels == 4:
        image, _ = convert_rgba2bgra(image)
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY), None
    elif channels == 3:
        image, _ = convert_rgba2bgra(image)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None
    else:
        return image, None


def convert_bgra2rgba(image: np.array) ->Tuple[np.array, None]:
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Convert BGR(alpha) image to RGB(alpha) image

    PARAMETERS:
    -----------

    :param image (np.array): image to convert

    RETURN:
    -------

    :return image(np.array): a RGB(alpha) image
    """

    # Get the number of channels in the image
    _, _, channels = image.shape

    # Handle BGR and BGR(A) images
    if channels == 3:
        image = image[:, :, (2, 1, 0)]
    elif channels == 4:
        image = image[:, :, (2, 1, 0, 3)]
    return image, None


def convert_rgba2bgra(image) ->Tuple[np.array, None]:
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

    # Return the transform image and None
    return image, None


def remove_channel(image: np.array, index_channel: int)->Tuple[np.array, None]:
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

    :return (np.array): The image without the selected channel
    :return: None
    """
    return np.delete(image, index_channel, -1), None

"""
"
" SEMANTIC SEGMENTATION
"
"""


def color2label(image: np.array, dict_labels: OrderedDict) -> Tuple[np.array, None]:
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
    :return: None
    """
    # Convert float to integer
    image = image.astype(np.uint64)

    # Get image shape
    h, w, c = image.shape

    # Initialize a label matrix
    labels = np.zeros((h, w, 1), dtype=int)

    # Get the list of colors
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


def label2color(labels: np.array, dict_labels: OrderedDict) -> Tuple[np.array, None]:
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Transform an array of labels to the corresponding RGB colors

    PARAMETERS:
    -----------

    :param labels (np.array): the array containing the labels
    :param dict_labels (OrderedDict): An dictionary containing the relation class name => color (e.g. "cat" : [250, 89, 52]

    RETURN:
    -------

    :return image (np.array): The image with the colors corresponding to the given labels
    :return: None
    """
    # Convert float to integer
    labels = labels.astype(np.uint64)

    # Get image shape
    h, w, _ = labels.shape

    # Initialize an image
    image = np.zeros((h, w, 3), dtype=np.uint8)

    # Get the color values
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