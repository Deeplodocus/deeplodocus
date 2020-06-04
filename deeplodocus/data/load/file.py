# Python imports
import os
from typing import Union
from typing import Any
from typing import Tuple
from typing import Optional

# Deeplodocus imports
from deeplodocus.utils.notification import Notification
from deeplodocus.utils.file import get_specific_line
from deeplodocus.data.load.source import Source

# Deeplodocus flags
from deeplodocus.flags.notif import *


class File(Source):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Load a source file.

    """

    def __init__(self,
                 index: int = -1,
                 is_loaded: bool = False,
                 is_transformed: bool = False,
                 path: str = "",
                 join: Optional[str] = None,
                 delimiter: str = ",",
                 num_instances=None,
                 instance_id: int = 0
                 ):

        super().__init__(index=index,
                         num_instances=num_instances,
                         is_loaded=is_loaded,
                         is_transformed=is_transformed,
                         instance_id=instance_id)

        self.path = path
        self.join = join
        self.delimiter = delimiter

    def __getitem__(self, index: int) -> Tuple[Any, bool, bool]:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the item at the selected index

        PARAMETERS:
        -----------

        :param index(int): The index of the selected item

        RETURN:
        -------

        :return data:
        """

        # Check wether the data is loaded and transformed
        is_loaded = self.is_loaded
        is_transformed = self.is_transformed

        # Get the data
        data = get_specific_line(filename=self.path,
                                 index=index)

        # create a sequence if necessary
        if isinstance(data, str):
            data = data.split(self.delimiter)  # Generate a list from the sequence
            if len(data) == 1:
                data = data[0]

        # Format the data if it is a path to a specific file
        # Formatting will automatically join the adequate parent directory to a relative path
        if self.join is not None:
            data = self.__format_path(data)

        return data, is_loaded, is_transformed

    def __len__(self) -> int:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Calculate the length of the source

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return length(int): The length of the source
        """
        return self.num_instances

    def compute_length(self) -> int:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Compute the length using the method corresponding to the source type

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return length(int): The length of the source
        """
        length = 0

        with open(self.path) as f:
            for _ in f:
                length += 1

        return length

    def __format_path(self, data: Union[str, list]) -> Union[list, str]:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Format the data if it is a path

        PARAMETERS:
        -----------
        :param data (str): The initial data in a string format

        RETURN:
        -------

        :return data(str): The data formatted to join the
        """
        if isinstance(data, list):
            for i, d in enumerate(data):
                data[i] = self.__format_path(d)
        else:
            if self.join.lower() == "auto":
                data = "/".join([os.path.dirname(self.path), data])
            elif os.path.isdir(self.join):
                data = "/".join([self.join, data])
            else:
                Notification(DEEP_NOTIF_FATAL,
                             "The following folder couldn't be joined to the file path : %s" % str(self.join),
                             solutions="Make sure the folder to join exists.")
        return data

    def check(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------


        :return:
        """

        super().check()

        if not os.path.isfile(self.path):
            Notification(DEEP_NOTIF_FATAL, "The following path is not a source file : " + str(self.path))
        else:
            Notification(DEEP_NOTIF_SUCCESS, "Source file \"%s\" successfully loaded" % self.path)
