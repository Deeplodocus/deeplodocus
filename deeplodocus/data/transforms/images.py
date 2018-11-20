import random
import numpy as np
import cv2


def random_blur(image, kernel_size_min, kernel_size_max):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Apply a random blur to the image


    PARAMETERS:
    -----------

    :param image:
    :param kernel_size_min:
    :param kernel_size_max:


    RETURN:
    -------

    :return: The image and the last transform data
    """
    kernel_size = (random.randint(kernel_size_min // 2, kernel_size_max // 2)) * 2 + 1
    image, _ = blur(image, kernel_size)
    transform = ["blur", blur, {"kernel_size": kernel_size}]
    return image, transform


def blur(image, kernel_size):
    kernel = (int(kernel_size), int(kernel_size))
    image = cv2.blur(image, kernel)
    return image, None


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)
