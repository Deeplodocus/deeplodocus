# Python imports
import numpy as np
import random
from typing import Optional
from typing import Tuple
from typing import Union
from typing import List

# Third party libs
import cv2

# Deeplodocus imports
from deeplodocus.utils.generic_utils import get_module
from deeplodocus.utils.notification import Notification
from deeplodocus.utils.flag import Flag
from deeplodocus.utils.namespace import Namespace

# Deeplodocus flags
from deeplodocus.utils.flags.module import DEEP_MODULE_TRANSFORMS
from deeplodocus.utils.flags.notif import DEEP_NOTIF_INFO, DEEP_NOTIF_FATAL


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

    def __init__(self, name, mandatory_transforms : Union[Namespace, List[dict]], transforms: Union[Namespace, List[dict]]):
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
        self.__name = name
        self.last_index = None
        self.transformer_entry = None
        self.transformer_index = None
        self.last_transforms = []

        # List of transforms
        self.list_transforms = self.__fill_transform_list(transforms)
        self.list_mandatory_transforms = self.__fill_transform_list(mandatory_transforms)

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

        Notification(DEEP_NOTIF_INFO, "Transformer '" + str(self.__name) + "' summary :")

        # MANDATORY TRANSFORMS
        if len(self.list_mandatory_transforms) > 0:
            Notification(DEEP_NOTIF_INFO, " Mandatory transforms :")
            for t in self.list_mandatory_transforms:
                Notification(DEEP_NOTIF_INFO, "--> Name : " + str(t["name"]) + " , Args : " + str(t["kwargs"]) + ", Method : " + str(t["method"]))

        # TRANSFORMS
        if len(self.list_transforms) > 0:
            Notification(DEEP_NOTIF_INFO, " Transforms :")
            for t in self.list_transforms:
                Notification(DEEP_NOTIF_INFO, "--> Name : " + str(t["name"]) + " , Args : " + str(t["kwargs"]) + ", Method : " + str(t["method"]))

    def get_pointer(self) -> Tuple[Optional[Flag], Optional[int]]:
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
        return self.transformer_entry, self.transformer_index

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

    @staticmethod
    def __fill_transform_list(transforms: Union[Namespace, List[dict]]) -> list:
        """
        AUTHORS:
        --------

        author: Alix Leroy

        DESCRIPTION:
        ------------

        Fill the list of transforms with the corresponding methods and arguments

        PARAMETERS:
        -----------

        :param transforms (Union[Namespace, list): A list of transforms

        RETURN:
        -------

        :return loaded_transforms (list): The list of loaded transforms
        """

        loaded_transforms = []
        if transforms is not None:
            for transform in transforms:
                if "module" not in transform:
                    transform["module"] = None

                loaded_transforms.append({"name": transform["name"],
                                          "method": get_module(name=transform["name"],
                                                               module=transform["module"],
                                                               browse=DEEP_MODULE_TRANSFORMS),
                                          "kwargs": transform["kwargs"]})
        return loaded_transforms

    def transform(self, data, index, augment: bool):
        """
        Authors : Alix Leroy,
        :param data: data to transform
        :param index: The index of the instance in the Data Frame
        :param augment: bool:
        :return: The transformed data
        """
        pass # Will be overridden

    def apply_transforms(self, transformed_data, transforms):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Apply the list of transforms to the data

        PARAMETERS:
        -----------

        :param transformed_data:
        :param transforms:

        RETURN:
        -------

        :return transformed_data: The transformed data
        """
        # Apply the transforms
        for transform in transforms:
            transform_name = transform["name"]
            transform_method = transform["method"]  # Create a generic alias for the transform method
            transform_args = transform["kwargs"]  # Dictionary of arguments
            transformed_data, last_method_used = transform_method(transformed_data, **transform_args)       # Apply the transform

            # Update the last transforms used and the last index
            if last_method_used is None:
                self.last_transforms.append([transform_name, transform_method, transform_args])

            else:
                self.last_transforms.append(last_method_used)
        return transformed_data

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

    @staticmethod
    def has_transforms():
        return True
