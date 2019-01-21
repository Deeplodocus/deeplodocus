import os
import shutil
import datetime

from deeplodocus.utils.flags.ext import DEEP_EXT_CSV


class Logs(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy and Samuel Westlake

    DESCRIPTION:
    ------------

    A class which manages the logs
    """

    def __init__(self, log_type: str,
                 directory: str = "%s/logs",
                 extension: str = DEEP_EXT_CSV,
                 write_time=True) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy and Samuel Westlake

        DESCRIPTION:
        ------------

        Initialize a log object.

        PARAMETERS:
        -----------

        :param log_type: str: The log type (notification, history
        :param directory: str
        :param extension: str:
        :param write_time: bool: Whether or not to start the line with a time stamp

        RETURN:
        -------

        :return: None
        """
        self.log_type = log_type
        self.directory = directory
        self.extension = extension
        self.write_time = write_time
        self.__check_exists()

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

        :param text: str: The text to add
        :param write_time: Whether or now to name the file with a time stamp

        RETURN:
        -------

        :return: None

        """
        self.__check_exists()
        file_path = self.__get_path()
        time_str = datetime.datetime.now() if write_time else ""
        with open(file_path, "a") as log:
            log.write("%s : %s\n" % (time_str, text))

    def delete(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy and Samuel Westlake

        DESCRIPTION:
        ------------

        Delete the log file.

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None

        """
        try:
            os.remove(self.__get_path())
        except FileNotFoundError:
            pass

    def close(self, new_directory=None):
        """
        AUTHORS:
        --------

        :author: Alix Leroy and Samuel Westlake

        DESCRIPTION:
        ------------

        Close the log file by renaming it to include a timestamp from the last line

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None

        """
        # We need a timestamp to give the log file a unique name.
        # The timestamp from the last line of the log file is preferred over datetime.now() ...
        # because we may be cleaning up and closing an old logfile from a previous, interrupted run.
        time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        old_path = self.__get_path()
        self.directory = new_directory
        os.makedirs(self.directory, exist_ok=True)
        new_path = self.__get_path(time)
        shutil.move(old_path, new_path)

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
        None

        RETURN:
        -------

        :return: None
        """
        if not os.path.isfile(self.__get_path()):
            os.makedirs(self.directory, exist_ok=True)
            open(self.__get_path(), "w").close()

    def __get_path(self, time=None):
        """
        :return:
        """
        if time is None:
            return "%s/%s%s" % (self.directory, self.log_type, self.extension)
        else:
            return "%s/%s_%s%s" % (self.directory, self.log_type, time, self.extension)
