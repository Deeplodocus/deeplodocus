import cv2
import numpy as np
import random
from typing import Union
import pkgutil

import __main__

import deeplodocus.data.transforms as tfm
from deeplodocus.utils.namespace import Namespace
from deeplodocus.utils.notification import Notification
from deeplodocus.utils.flags import *
from deeplodocus.utils import get_main_path


class Transformer(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy
    :author: Samuel Westlake

    DESCRIPTION:
    ------------

    A generic transformer class.
    The transformer loads the transforms in memory and allows the data to be transformed
    """

    def __init__(self, config):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize the Transformer by filling the transforms list

        PARAMETERS:
        -----------

        :param config->Namespace: The Namespace containing the config

        RETURN:
        -------

        :return: None
        """
        self.name = config.name
        self.last_index = None
        self.pointer_to_transformer = None
        self.last_transforms = []
        self.list_transforms = []
        self.__fill_transform_list(config.transforms)

    def summary(self):
        """
        AUTHORS:
        --------

        author: Alix Leroy

        DESCRIPTION:
        -----------

        Print the summary of the tranformer

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """

        Notification(DEEP_NOTIF_INFO, "Transformer '" + str(self.name) + "' summary :")

        for t in self.list_transforms:
            Notification(DEEP_NOTIF_INFO, "--> Name : " + str(t[0]) + " , Args : " + str(t[2]) + ", Method : " + str(t[1]))

    def get_pointer(self):
        """
        AUTHORS:
        --------

        author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the pointer to the other transformer

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: pointer_to_transformer attribute
        """
        return self.pointer_to_transformer

    def reset(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Reset the last index and last transforms used after one epoch

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """

        self.last_index = None
        self.last_transforms = []

    def __fill_transform_list(self, transforms):
        """
        AUTHORS:
        --------

        author: Alix Leroy

        DESCRIPTION:
        ------------

        Fill the list of transforms with the corresponding methods and arguments

        PARAMETERS:
        -----------

        :param transforms-> list: A list of transforms

        RETURN:
        -------

        :return: None
        """

        for i, transform in enumerate(transforms):
            key = list(transform.keys())[0]
            values = list(transform.values())[0]

            self.list_transforms.append([key, self.__get_transform(key), values])

    def __get_transform(self, transform_name: str)->callable:
        """
        AUTHORS:
        --------

        author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the transform method of an existing default one

        PARAMETERS:
        -----------

        :param transform-> str: A transform name

        RETURN:
        -------

        :return->callable: A callable method of the Transformer class
        """

        local = {"transform": None}

        # Get the transform method among the default ones
        for importer, modname, ispkg in pkgutil.walk_packages(path=tfm.__path__,
                                                              prefix=tfm.__name__ + '.',
                                                              onerror=lambda x: None):
            try:
                exec("from {0} import {1} \ntransform= {2}".format(modname, transform_name, transform_name), {}, local)
                break
            except:
                pass

        # Get the transform method among the custom ones
        if local["transform"] is None:
            for importer, modname, ispkg in pkgutil.walk_packages(path=[get_main_path() + "/modules/transforms"],
                                                                  prefix = "modules.transforms.",
                                                                  onerror=lambda x: None):
                try:
                    exec("from {0} import {1} \ntransform= {2}".format(modname, transform_name, transform_name), {}, local)
                    break
                except:
                    pass

        # If neither a standard not a custom transform is loaded
        if local["transform"] is None:
            Notification(DEEP_NOTIF_FATAL, "The following transform could not be loaded neither from the standard transforms nor from the custom ones : " + str(transform_name))

        return local["transform"]
        #return getattr(self, transform_name)



    def transform(self, data, index, data_type):
        """
        Authors : Alix Leroy,
        :param data: data to transform
        :param index: The index of the instance in the Data Frame
        :param data_type: The type of data
        :return: The transformed data
        """
        pass # Will be overridden




    def __transform_image(self, image, key):

        """
        Author : Alix Leroy
        :param image: input image to augment
        :param key: the parameters of the augmentation in a dictionnary
        :return: augmented image
        """

        ################
        # ILLUMINATION #
        ################
        if key == "adjust_gamma":
            gamma = np.random.random(key["gamma"][0], key["gamma"][1])
            image = self.adjust_gamma(image, gamma)


        #########
        # BLURS #
        #########
        elif key == "average":
            kernel = tuple(int(key["kernel_size"]), int(key["kernel_size"]))
            image = cv2.blur(image, kernel)

        elif key == "gaussian_blur":
            kernel = tuple(int(key["kernel_size"]), int(key["kernel_size"]))
            image = cv2.GaussianBlur(image, kernel, 0)

        elif key == "median_blur":

            image = cv2.medianBlur(image, int(key["kernel_size"]))

        elif key == "bilateral_blur":
            diameter = int(key["diameter"])
            sigma_color = int(key["sigma_color"])
            sigma_space = int(key["sigma_space"])
            image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


        #########
        # FLIPS #
        #########
        elif key == "horizontal_flip":
            image = cv2.flip(image, 0)

        elif key == "vertical_flip":
            image = cv2.flip(image, 1)


        #############
        # ROTATIONS #
        #############

        elif key == "random_rotation":
            angle = np.random.random(00, 359.9)
            shape = image.shape
            rows, cols = shape[0:2]
            m = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            image = cv2.warpAffine(image, m, (cols, rows)).astype(np.float32)


        elif key == "boundary_rotation":
            angle = float(key["angle"])
            angle = (2 * np.random.rand() - 1) * angle
            shape = image.shape
            rows, cols = shape[0:2]
            m = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            image = cv2.warpAffine(image, m, (cols, rows)).astype(np.float32)


        elif key == "rotation":
            angle = float(key["angle"])
            shape = image.shape
            rows, cols = shape[0:2]
            m = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            image = cv2.warpAffine(image, m, (cols, rows)).astype(np.float32)
        else:
            Notification(DEEP_NOTIF_FATAL, "This transformation function does not exist : " + str(transformation))
        return image

    def random_blur(self, image, kernel_size_min, kernel_size_max):

        kernel_size = (random.randint(kernel_size_min//2, kernel_size_max//2)) * 2 + 1
        image, _ = self.blur(image, kernel_size)
        transform =  ["blur", self.blur, {"kernel_size": kernel_size}]
        return image, transform

    def blur(self, image, kernel_size):
        #kernel = tuple(int(kernel_size), int(kernel_size))
        kernel = (int(kernel_size), int(kernel_size))
        image = cv2.blur(image, kernel)
        return image, None




    def normalize_video(self, video):
        """
        Author: Alix Leroy
        :param video: sequence of frames
        :return: a normalized sequence of frames
        """

        video = [self.normalize_image(frame) for frame in video]

        return video

