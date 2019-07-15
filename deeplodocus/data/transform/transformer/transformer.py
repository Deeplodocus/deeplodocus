# Python imports
import numpy as np
import random
from typing import Optional
from typing import Tuple
from typing import Union
from typing import List
from typing import Any

# Third party libs
import cv2

# Deeplodocus imports
from deeplodocus.utils.generic_utils import get_module
from deeplodocus.utils.notification import Notification
from deeplodocus.utils.flag import Flag
from deeplodocus.utils.namespace import Namespace
from deeplodocus.utils.flags.msg import DEEP_MSG_TRANSFORM_VALUE_ERROR
from deeplodocus.data.transform.transform_data import TransformData

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

    def __init__(self,
                 name: str,
                 mandatory_transforms_start: Union[Namespace, List[dict]],
                 transforms: Union[Namespace, List[dict]],
                 mandatory_transforms_end: Union[Namespace, List[dict]]):
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
        self.list_mandatory_transforms_start = self.__fill_transform_list(mandatory_transforms_start)
        self.list_mandatory_transforms_end = self.__fill_transform_list(mandatory_transforms_end)



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

        # MANDATORY TRANSFORMS START
        if len(self.list_mandatory_transforms_start) > 0:
            Notification(DEEP_NOTIF_INFO, " Mandatory transforms at start:")
            for t in self.list_mandatory_transforms_start:
                Notification(DEEP_NOTIF_INFO, "--> Name : " + str(t.name) + " , Args : " + str(t.kwargs) + ", Module path: " + str(t.module_path))

        # TRANSFORMS
        if len(self.list_transforms) > 0:
            Notification(DEEP_NOTIF_INFO, " Transforms :")
            for t in self.list_transforms:
                Notification(DEEP_NOTIF_INFO, "--> Name : " + str(t.name) + " , Args : " + str(t.kwargs) + ", Module path: " + str(t.module_path))


        # MANDATORY TRANSFORMS END
        if len(self.list_mandatory_transforms_end) > 0:
            Notification(DEEP_NOTIF_INFO, " Mandatory transforms at end:")
            for t in self.list_mandatory_transforms_end:
                Notification(DEEP_NOTIF_INFO, "--> Name : " + str(t.name) + " , Args : " + str(t.kwargs) + ", Module path: " + str(t.module_path))

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

                module, module_path = get_module(
                    name=transform["name"],
                    module=transform["module"],
                    browse=DEEP_MODULE_TRANSFORMS
                )

                t = TransformData(name=transform["name"],
                                  method=module,
                                  module_path=module_path,
                                  kwargs=transform["kwargs"]
                                  )
                loaded_transforms.append(t)

        return loaded_transforms

    def transform(self, data: Any, index: int, augment: bool)-> Any:
        """
        Authors : Alix Leroy,
        :param data: data to transform
        :param index: The index of the instance in the Data Frame
        :param augment: bool:
        :return: The transformed data
        """
        pass # Will be overridden

    def apply_transforms(self, transformed_data, transforms: List[TransformData]) -> Any:
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

            try:
                # Transform the data and get the transformation settings if a random transform is applied (None else)
                transformed_data, last_method_used = transform.method(transformed_data, **transform.kwargs)
            except ValueError as e:
                Notification(
                    DEEP_NOTIF_FATAL,
                    "ValueError : %s : %s" % (str(e), DEEP_MSG_TRANSFORM_VALUE_ERROR)
                )
            except TypeError as e:
                Notification(
                    DEEP_NOTIF_FATAL,
                    "TypeError : %s" % (str(e))
                )

            # Update the last transforms used and the last index
            if last_method_used is None:
                self.last_transforms.append(transform)

            else:
                self.last_transforms.append(last_method_used)
        return transformed_data

    @staticmethod
    def has_transforms():
        return True
