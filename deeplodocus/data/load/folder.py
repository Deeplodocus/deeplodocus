# Python imports
import os
from typing import Any
from typing import Tuple

# Deeplodocus imports
from deeplodocus.utils.notification import Notification
from deeplodocus.utils.file import get_specific_line

# Deeplodocus flags
from deeplodocus.flags.notif import *
from deeplodocus.data.load.source import Source


class Folder(Source):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Load a source folder.

    """

    def __init__(self,
                 id=-1,
                 is_loaded=False,
                 is_transformed=False,
                 path="",
                 join=None,
                 delimiter=",",
                 end_line="\n",
                 num_instances=None
                 ):

        super().__init__(id, is_loaded, is_transformed)

        self.path = path
        self.join = join
        self.delimiter = delimiter
        self.end_line = end_line


        if num_instances is None:
            self.num_instances = self.__compute_length_()
        else:
            self.num_instances = num_instances

        # check if the source is a real file
        self.__check_source()

    """
    "
    " LOAD ITEM
    "
    """

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
        data = get_specific_line(filename=self.source,
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

        with open(self.source) as f:
            for l in f:
                length += 1

        return length


    def __check_source(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------


        :return:
        """

        if not os.path.isfile(self.path):
            Notification(DEEP_NOTIF_FATAL, "The following path is not a source file : " + str(self.path))
        else:
            Notification(DEEP_NOTIF_SUCCESS, "Source file %i successfully loaded".format(self.id))

    def __generate_file_equivalent(self):

        filepath = ""
        num_instances =

        file = File(id=self.id,
                    is_loaded=self.is_loaded,
                    is_transformed=self.is_transformed,
                    path=filepath,
                    join=self.join,
                    delimiter=",",
                    end_line="\n",
                    num_instances=num_instances
                    )

        return file