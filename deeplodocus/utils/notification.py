from deeplodocus.utils import get_main_path
from deeplodocus.utils.colors import *
from deeplodocus.utils.deep_error import DeepError
from deeplodocus.utils.flags.ext import DEEP_EXT_LOGS
from deeplodocus.utils.flags.log import DEEP_LOG_NOTIFICATION
from deeplodocus.utils.flags.msg import DEEP_MSG_NOTIF_UNKNOWN
from deeplodocus.utils.flags.notif import *
from deeplodocus.utils.logs import Logs
from deeplodocus.utils.flag import Flag


class Notification(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy
    :author: Samuel Westlake

    DESCRIPTION:
    ------------

    Display a custom message to the user and save it to the logs if required.

    """

    def __init__(self, notif_flag: Flag, message: str, log: bool = True, solutions=None) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Check what type of notification has to be displayed.
        Call the appropriate private method to write the notification as intended.

        PARAMETERS:
        -----------

        :param notif_type (int): Index of the notification type flag.
        :param message (str): Message to display.
        :param log (bool): Whether or not to write message to log file.

        RETURN:
        -------

        :return : None

        """
        self.log = log                          # Whether or not notifications should be written to logs
        self.response = ""                      # Allocated by self.__input(), returned by self.get()

        if isinstance(notif_flag, Flag):
            # INFO
            if DEEP_NOTIF_INFO.corresponds(notif_flag):
                self.__info(message)

            # DEBUG
            elif DEEP_NOTIF_DEBUG.corresponds(notif_flag):
                self.__debug(message)

            # SUCCESS
            elif DEEP_NOTIF_SUCCESS.corresponds(notif_flag):
                self.__success(message)

            # WARNING
            elif DEEP_NOTIF_WARNING.corresponds(notif_flag):
                self.__warning(message)

            # ERROR
            elif DEEP_NOTIF_ERROR.corresponds(notif_flag):
                self.__error(message)

            # FATAL
            elif DEEP_NOTIF_FATAL.corresponds(notif_flag):
                self.__fatal_error(message, solutions=solutions)

            # INPUT
            elif DEEP_NOTIF_INPUT.corresponds(notif_flag):
                self.__input(message)

            # RESULT
            elif DEEP_NOTIF_RESULT.corresponds(notif_flag):
                self.__result(message)

            # LOVE
            elif DEEP_NOTIF_LOVE.corresponds(notif_flag):
                self.__love(message)

            # WRONG FLAG
            else:
                Notification(DEEP_NOTIF_FATAL, DEEP_MSG_NOTIF_UNKNOWN % notif_flag)

        # WRONG notification_type
        else:
            Notification(DEEP_NOTIF_FATAL, DEEP_MSG_NOTIF_UNKNOWN % notif_flag)

    def get(self) -> str:
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Get the result of the input.

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return self.response (str): The response given by the user.

        """
        return self.response

    def __fatal_error(self, message: str, solutions=None) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Display the given message with a RED background, preceded by 'DEEP FATAL : '.
        Write to log file if required.
        Raise a DeepError.

        PARAMETERS:
        -----------

        :param message (str): The message to display.

        RETURN:
        -------

        :return: None

        """
        # Print deep fatal errror
        message = "DEEP FATAL ERROR : %s" % message
        print("%s%s%s" % (CREDBG, message, CEND))
        if self.log is True:
            self.__add_log(message)

        # If possible solutions are given, print them too
        if solutions is not None:
            message = "DEEP INFO : %s" % "Possible solutions : "
            print("%s%s%s" % (CBLUE, message, CEND))
            if self.log is True:
                self.__add_log(message)
            solutions = solutions if isinstance(solutions, list) else [solutions]
            for i, solution in enumerate(solutions):
                message = "DEEP INFO : %i : %s" % (i + 1, solution)
                print("%s%s%s" % (CBLUE, message, CEND))
                if self.log is True:
                    self.__add_log(message)

        raise DeepError

    def __error(self, message: str) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :auhtor: Samuel Westlake

        DESCRIPTION:
        ------------

        Display the given message in RED, preceded by 'DEEP ERROR : '.
        Write to log file if required.

        PARAMETERS:
        -----------

        :param message (str): The message to display.

        RETURN:
        -------

        :return: None

        """
        message = "DEEP ERROR : %s" % message
        print("%s%s%s" % (CRED, message, CEND))
        if self.log is True:
            self.__add_log(message)

    def __warning(self, message: str) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Display the given message in ORANGE/YELLOW, preceded by 'DEEP WARNING : '.
        Write to log file if required.

        PARAMETERS:
        -----------

        :param message (str): The message to display.

        RETURN:
        -------

        :return: None

        """
        message = "DEEP WARNING : %s" % message
        print("%s%s%s" % (CYELLOW2, message, CEND))
        if self.log is True:
            self.__add_log(message)

    def __debug(self, message: str) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Display the given message in BEIGE, preceded by 'DEEP DEBUG : '.
        Write to log file if required.

        PARAMETERS:
        -----------

        :param message (str): The message to display.

        RETURN:
        -------

        :return: None

        """
        message = "DEEP DEBUG : %s" % message
        print("%s%s%s" % (CBEIGE, message, CEND))
        if self.log is True:
            self.__add_log(message)

    def __success(self, message: str) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Display the given message in GREEM, preceded by 'DEEP SUCCESS : '.
        Write to log file if required.

        PARAMETERS:
        -----------

        :param message (str): The message to display.

        RETURN:
        -------

        :return: None

        """
        message = "DEEP SUCCESS : %s" % message
        print("%s%s%s" % (CGREEN, message, CEND))
        if self.log is True:
            self.__add_log(message)

    def __info(self, message: str) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Display the given message in BLUE, preceded by 'DEEP INFO : '.
        Write to log file if required.

        PARAMETERS:
        -----------

        :param message (str): The message to display.

        RETURN:
        -------

        :return: None

        """
        message = "DEEP INFO : %s" % message
        print("%s%s%s" % (CBLUE, message, CEND))
        if self.log is True:
            self.__add_log(message)

    def __result(self, message: str) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Display the given message in WHITE preceded by 'DEEP RESULT : '.
        Write to log file if required.

        PARAMETERS:
        -----------

        :param message (str): The message to display.

        RETURN:
        -------

        :return: None

        """
        message = "DEEP RESULT : %s" % message
        print(message)
        if self.log is True:
            self.__add_log(message)

    def __love(self, message: str) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Display the given message in PINK preceded by 'DEEP LOVE : '.
        Write to log file if required.

        PARAMETERS:
        -----------

        :param message (str): The message to display.

        RETURN:
        -------

        :return: None

        """
        message = "DEEP LOVE : %s" % message
        print("%s%s%s" % (CVIOLET2, message, CEND))
        if self.log is True:
            self.__add_log(message)

    def __input(self, message: str) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Display a BLINKING WHITE message and await for an input.

        PARAMETERS:
        -----------

        :param message (str): The message to display.

        RETURN:
        -------

        :return: None
        """
        message = "DEEP INPUT : " + str(message)
        print(CBLINK + CBOLD + str(message) + CEND)
        # Wait for an input from the user
        self.response = input("> ")
        if self.log is True:
            # Add the the message to the log
            self.__add_log(message)
            self.__add_log(str(self.response))

    @staticmethod
    def __add_log(message: str) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Add a message to the logs.

        PARAMETERS:
        -----------

        :param message (str): The message to save in the logs.

        RETURN:
        -------

        :return: None

        """
        Logs(
            log_type=DEEP_LOG_NOTIFICATION,
            directory=get_main_path(),
            extension=DEEP_EXT_LOGS
        ).add(message)
