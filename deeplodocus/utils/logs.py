import os
import shutil
import datetime
import __main__

from deeplodocus.utils.flags import *


class Logs(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    A class which manages the logs
    """

    def __init__(self, type: str,
                 directory: str ="%s/logs" % os.path.dirname(os.path.abspath(__main__.__file__)),
                 extension: str = ".csv",
                 write_time=True) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize a log object

        PARAMETERS:
        -----------

        :param type->str: The log type

        RETURN:
        -------

        :return: None
        """
        self.type = type
        self.directory = directory
        self.extension = extension
        self.write_time = write_time
        self.init_time = None
        self.__check_exists()

    def delete(self):
        """
        :return:
        """
        try:
            os.remove(self.__get_path())
        except FileNotFoundError:
            pass

    def add(self, text: str, write_time=True) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy and SW

        DESCRIPTION:
        ------------

        Add a line to the log

        PARAMETERS:
        -----------

        :param text -> str: The text to add

        RETURN:
        -------

        :return: None

        """
        self.__check_exists()
        file_path = self.__get_path()
        if write_time is True:
            time_str = datetime.datetime.now()
        else:
            time_str = ""
        with open(file_path, "a") as log:
            log.write("%s : %s\n" % (time_str, text))

    def __check_exists(self) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy and SW

        DESCRIPTION:
        ------------

        Create the log file and insert the date time on first line

        PARAMETERS:
        -----------
        :param logs_path -> str: The path to the log file

        RETURN:
        -------

        :return: None
        """
        if not os.path.isfile(self.__get_path()):
            os.makedirs(self.directory, exist_ok=True)
            open(self.__get_path(), "w").close()

    def close(self):
        """
        :return:
        """
        old_path = self.__get_path()
        self.init_time = datetime.datetime.now().strftime(TIME_FORMAT)
        shutil.move(old_path, self.__get_path())

    def __get_path(self):
        """
        :return:
        """
        if self.init_time is None:
            return "%s/%s%s" % (self.directory, self.type, self.extension)
        else:
            return "%s/%s_%s%s" % (self.directory, self.type, self.init_time, self.extension)
