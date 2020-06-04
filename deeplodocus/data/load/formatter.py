# Python imports
from typing import Optional
from typing import List
from typing import Any
from typing import Union
import numpy as np
import weakref

# Deeplodocus imports
from  deeplodocus.utils.notification import Notification
from deeplodocus.utils.generic_utils import get_corresponding_flag

# Deeplodocus flags
from deeplodocus.flags import *


class Formatter(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Format the data
            1) Format the data type
            2) Move the axis to a define sequence
    """


    #TODO: Put in Entry

    def __init__(self,
                 pipeline_entry: weakref,
                 move_axis: Optional[List[int]] = None,
                 convert_to: Optional[List[int]] = None,
                 ):

        # Pipeline entry weakref
        self.pipeline_entry = pipeline_entry

        # Optional convert_to attribute to define final data type
        self.convert_to = self.__check_convert_to(convert_to)

        # Optional sequence to move_axis
        self.move_axis = self.__check_move_axis(move_axis)

    def format(self, data: Any, entry_type: Flag) -> Any:
        """
        AUTHORS:
        --------

        :author: Samuel Westlake
        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Method to format data to PyTorch conventions.
        Images are converted from (w, h, ch) to (ch, h, w)
        Videos are ...

        PARAMETERS:
        -----------

        :param data (Any): the data to format

        RETURN:
        -------

        :return data (Any): the formatted data entry
        """
        # If we have to format a list of items
        if isinstance(data, list):
            formatted_data = []

            for d in data:

                fd = self.format(d, entry_type)
                formatted_data.append(fd)
            data = formatted_data

        # Else it is a unique item
        else:

            # Convert data type
            if self.convert_to is not None:
                # Use the name of the Flag instance directly as the name of the data type
                data = data.astype(self.convert_to.names[0])

            # Move the axes
            if self.move_axis is not None:
                data = self.__transpose(data)

        return data

    def __check_move_axis(self, move_axis: Optional[List[int]]) -> Optional[List[int]]:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check that the move_axis argument is a list of integer or None

        PARAMETERS:
        -----------

        :param move_axis (Optional[List[int]]): The new order of axis

        RETURN:
        -------
        :return (Optional[List[int]]):
        """
        # Check if None
        if move_axis is None:
            return None

        if isinstance(move_axis, list):
            for i in move_axis:
                if isinstance(i, int) is False:

                    # Get index of the PipelineEntry instance (throught its weakref)
                    entry_info = self.pipeline_entry().get_index()

                    # Get info of the Dataset (throught its weakref) from the weakref of the PipelineEntry instance
                    dataset_info = self.pipeline_entry().get_dataset()().get_info()

                    Notification(DEEP_NOTIF_FATAL,
                                 "Please check the value of the move_axis argument. in the Entry %i of the Dataset %s" % (entry_info, dataset_info),
                                 solutions="One of the item in the list is not an integer")

            return move_axis
        else:
            # Get index of the PipelineEntry instance (throught its weakref)
            entry_info = self.pipeline_entry().get_index()

            # Get info of the Dataset (throught its weakref) from the weakref of the PipelineEntry instance
            dataset_info = self.pipeline_entry().get_dataset().get_info()

            Notification(DEEP_NOTIF_FATAL,
                         "Please check the value of the move_axis argument in the Entry %i of the Dataset %s" % (entry_info, dataset_info),
                         solutions="The move_axis argument must be a list of integers index at 0")

    @staticmethod
    def __check_convert_to(convert_to: Union[str, None, Flag]) -> Optional[Flag]:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check if the convert_to argument is correct and return the corresponding Flag

        PARAMETERS:
        -----------

        :param convert_to(Union[str, None, Flag]): The convert_to argument given in the config file

        RETURN:
        -------

        :return (Flag): The corresponding DEEP_DTYPE_AS flag.
        """
        if convert_to is None:
            return None
        else:
            return get_corresponding_flag(
                flag_list=DEEP_LIST_DTYPE,
                info=convert_to
            )

    def __transpose(self, data) -> np.array:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Move axes of the data

        PARAMETERS:
        -----------

        :param data (np.array): The data needing a axis swap
        :param move_axes(List[int]): The new axes order

        RETURN:
        -------

        :return data (np.array): The data with teh axes swapped
        """

        return np.transpose(data, self.move_axis)
