# Python imports
import os
from typing import List
from typing import Any
from typing import Tuple
from typing import Union

# Deeplodocus imports
from deeplodocus.utils.notification import Notification
from deeplodocus.utils.file import get_specific_line
from deeplodocus.flags import *


# Deeplodocus flags

class Source(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Source class
    This class contains all the information to access the data whatever the source given.
    The following sources are supported :
                                         - Files
                                         - Folders
                                         - Database (Work in progress)
                                         - Spark (Work in progress)
                                         - Specific server (work in progress)
    """

    def __init__(self, source: str, join: str):
        self.premade_dataset = None
        self.source = source
        self.type = self.__check_source_type(source)
        self.dataset = None
        self.join = join
        self.data_in_memory = None
        self.length = None
        self.delimiter = ";"

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

        is_loaded = False
        is_transformed = False

        # FILE
        if DEEP_SOURCE_FILE.corresponds(self.type):
            data = get_specific_line(filename=self.source,
                                     index=index)
            is_loaded = False
            is_transformed = False

        # BINARY FILE
        # elif self.type == DEEP_SOURCE_BINARY_FILE()
        # TODO: Add binary files

        # FOLDER
        elif DEEP_SOURCE_FOLDER.corresponds(self.type):
            Notification(DEEP_NOTIF_FATAL, "Load from hard drive with a source folder is supposed to "
                                           "be converted to a source file."
                                           "Please check the documentation to see how to use the Dataset class")
            is_loaded = False
            is_transformed = False

        # DATABASE
        elif DEEP_SOURCE_DATABASE.corresponds(self.type):
            Notification(DEEP_NOTIF_FATAL, "Load from hard drive with a source database not implemented yet")
            is_loaded = False
            is_transformed = False

        # PREMADE DATASET
        elif DEEP_SOURCE_PREMADE_DATASET.corresponds(self.type):
            data = self.premade_dataset["dataset"].__getitem__(index)
            is_loaded = True
            is_transformed = False

        if isinstance(data, str):
            data = data.split(self.delimiter)  # Generate a list from the sequence
            if len(data) == 1:
                data = data[0]

        # Format the data if it is a path to a specific file
        # Formatting will automatically join the adequate parent directory to a relative path

        if self.join is not None:
            data = self.__format_path(data)

        return data, is_loaded, is_transformed

    def __len__(self, load_method : Flag) -> int:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Calculate the length of the source

        PARAMETERS:
        -----------

        :param load_method(Flag): The loading method selected

        RETURN:
        -------

        :return length(int): The length of the source
        """

        length = 0

        # OFFLINE
        if DEEP_LOAD_METHOD_OFFLINE.corresponds(load_method):
            length = len(self.data_in_memory)

        #elif DEEP_LOAD_METHOD_SEMI_ONLINE.corresponds(load_method):
        #    # TODO: add semi online

        # ONLINE
        elif DEEP_LOAD_METHOD_ONLINE.corresponds(load_method):
            length = self.__compute_length_()

        # DEFAULT
        else:
            length = self.__compute_length_()

        self.length = length
        return length

    def __compute_length_(self) -> int:
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

        # FILE
        if DEEP_SOURCE_FILE.corresponds(self.type):
            with open(self.source) as f:
                for l in f:
                    length += 1

        # FOLDER
        elif DEEP_SOURCE_FOLDER.corresponds(self.type):
            Notification(DEEP_NOTIF_FATAL, "Calculation of the source length not implemented for the folder")

        # DATABASE
        elif DEEP_SOURCE_DATABASE.corresponds(self.type):
            Notification(DEEP_NOTIF_FATAL, "Calculation of the source length not implemented for the database")

        # SERVER
        elif DEEP_SOURCE_SERVER.corresponds(self.type):
            Notification(DEEP_NOTIF_FATAL, "Calculation of the source length not implemented for a remote server")

        # SPARK
        elif DEEP_SOURCE_SPARK.corresponds(self.type):
            Notification(DEEP_NOTIF_FATAL, "Calculation of the source length not implemented for spark")

        # PREMADE DATASET
        elif DEEP_SOURCE_PREMADE_DATASET.corresponds(self.type):
            Notification(DEEP_NOTIF_FATAL, "Calculation of the source length not implemented for premade datasets")

        # OTHERS
        else:
            Notification(DEEP_NOTIF_FATAL,
                         "The length of the source %s could not be computed because the type is not handled." % str(self.source))
        return length

    def load_offline(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:

        :return:
        """
        pass

    """
    "
    " MEMORY
    "
    """

    def add_item_to_memory(self, item: Any, index: int) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Add an item to memory

        PARAMETERS:
        -----------

        :param index (int): The index of the item
        :param item (Any): The item to add in memory

        RETURN:
        -------

        :return: None
        """

        self.data_in_memory[index] = item

    def initialize_memory(self, instances: int) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize the list storing the data

        PARAMETERS:
        -----------

        :param instances(int): The number of instances in the source

        RETURN:
        -------

        :return: None
        """

        self.data_in_memory = [0 for k in range(instances)]


    def __check_source_type(self, source: str) -> Flag:
        """
        AUTHORS:
        --------

        author: Alix Leroy

        DESCRIPTION:
        ------------

        Find the type of the source path

        PARAMETERS:
        -----------

        :param sources (List[str]): The list of sources

        RETURN:
        -------

        :return type (int): A type of flag
        """
        # FILE
        if os.path.isfile(source):
            return DEEP_SOURCE_FILE

        # FOLDER
        elif os.path.isdir(source):
            return DEEP_SOURCE_FOLDER

        # DATABASE
        elif self.__is_source_database(source):
            return DEEP_SOURCE_DATABASE

        # PRE-MADE DATASET
        elif self.__is_source_premade_dataset(source):
            return DEEP_SOURCE_PREMADE_DATASET

        else:
            Notification(DEEP_NOTIF_FATAL, DEEP_MSG_DATA_SOURCE_NOT_FOUND % source)


    """
    "
    " DATABASE
    "
    """

    def __list_available_databases(self):
        # TODO : List the databases in the config folder
        return []

    def __is_source_database(self, source):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check if the given source is a database referenced

        PARAMETERS:
        -----------

        :param source (str): The source string given by the user in the config file

        RETURN:
        -------

        :return (bool):  Whether the source is a database or not
        """

        if source[:4] == "db::":
            database = source.split("::")[1]
            if database in self.__list_available_databases():
                return True
            else:
                Notification(DEEP_NOTIF_FATAL, "The database '%s' is not recognized, please check the parameters given")
            pass
        else:
            return False

    def __is_source_premade_dataset(self, source: str) -> bool:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check if the source is a premade dataset

        PARAMETERS:
        -----------

        :param source (str): The source string given by the user in the config file

        RETURN:
        -------

        :return (bool): Whether the source is a database or not
        """

        Notification(DEEP_NOTIF_FATAL, "Loading from premade dataset not implemented yet.")


    """
    "
    " JOIN
    "
    """
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
                data = "/".join([os.path.dirname(self.source), data])
            elif os.path.isdir(self.join):
                data = "/".join([self.join, data])
            else:
                Notification(DEEP_NOTIF_FATAL, "The following folder couldn't be joined to the filepath : %s" % str(self.join))

        return data


    """
    "
    " GETTERS
    "
    """

    def get_source(self):
        return self.source

    def get_join(self):
        return self.join

    def get_type(self):
        return self.type

    def get_length(self):
        return self.length


    """
    "
    " SETTERS
    "
    """

    def set_source(self, source: str) -> None:
        self.source = source

    def set_type(self, type: Flag) -> None:
        self.type = type

    def set_data_in_memory(self, content: List[Any]) -> None:
        self.data_in_memory = content
