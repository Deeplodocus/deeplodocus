# Python imports
import os

# Deeplodocus imports
from deeplodocus.utils.notification import Notification

# Deeplodocus flags
from deeplodocus.utils.flags.source import *
from deeplodocus.utils.flags.notif import *
from deeplodocus.utils.flags.msg import *


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

    def __init__(self, source: str, join : str):
        self.source = source
        self.type = self.__check_source_type(source)
        self.join = join
        self.length = self.__calculate_length(source_type=self.type, source=source)

    """
    "
    " LOAD ITEM
    "
    """

    def __getitem__(self, index: int):
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

        data = self.__format_path(data)



    """
    "
    " LENGTH
    "
    """

    def __len__(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the length of the source

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return self.length(int): The length of the source
        """
        return self.length


    @staticmethod
    def __calculate_length(source_type : Flag, source: str):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Calculate the length of a specific source

        PARAMETERS:
        -----------

        :param source_type (Flag): The source type flag

        RETURN:
        -------

        :return length(int): The length of the source
        """

        # FILE
        if source_type() == DEEP_SOURCE_FILE():
            pass

        # FOLDER
        elif source_type() == DEEP_SOURCE_FOLDER():
            pass

        # DATABASE
        elif source_type() == DEEP_SOURCE_DATABASE():
            Notification(DEEP_NOTIF_FATAL, "Calculation of the source length not implemented for the database")

        # SERVER
        elif source_type() == DEEP_SOURCE_SERVER():
            Notification(DEEP_NOTIF_FATAL, "Calculation of the source length not implemented for a remote server")

        # SPARK
        elif source_type() == DEEP_SOURCE_SPARK():
            Notification(DEEP_NOTIF_FATAL, "Calculation of the source length not implemented for spark")

        # OTHERS
        else:
            Notification(DEEP_NOTIF_FATAL, "The length of the source %s could not be computed because the type is not handled." %str(source))

    def __check_source_type(self, source : str) -> Flag:
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

        :param f:

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


    """
    "
    " JOIN
    "
    """
    def __format_path(self, data: str):
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

        # If it is a file we format the path
        if os.path.isfile(data):
            if not os.path.isabs(data):
                if self.get_join() is not None:
                    if self.get_join().lower() == "auto":
                        data = "/".join([os.path.dirname(self.get_source()), data])
                    elif os.path.isdir(self.get_join()):
                        data = "/".join([os.path.dirname(self.get_join()), data])
                    else :
                        Notification(DEEP_NOTIF_FATAL, "The following folder couldn't be joined to the filepath : %s " % str(self.get_join()))
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

    """
    "
    " SETTERS
    "
    """

    def set_source(self, source : str):
        self.source = source

    def set_type(self, type : Flag):
        self.type = type