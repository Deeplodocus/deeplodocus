import os
import __main__

from deeplodocus.utils.logs import Logs
# from deeplodocus.utils.end import End
from deeplodocus.utils.flags import *


class DeepError(Exception):
    pass

#
# List of color codes
# Found at : https://stackoverflow.com/questions/287871/print-in-terminal-with-colors
#

CEND = '\33[0m'
CBOLD = '\33[1m'
CITALIC = '\33[3m'
CURL = '\33[4m'
CBLINK = '\33[5m'
CBLINK2 = '\33[6m'
CSELECTED = '\33[7m'

CBLACK = '\33[30m'
CRED = '\33[31m'
CGREEN = '\33[32m'
CYELLOW = '\33[33m'
CBLUE = '\33[34m'
CVIOLET = '\33[35m'
CBEIGE = '\33[36m'
CWHITE = '\33[37m'

CBLACKBG = '\33[40m'
CREDBG = '\33[41m'
CGREENBG = '\33[42m'
CYELLOWBG = '\33[43m'
CBLUEBG = '\33[44m'
CVIOLETBG = '\33[45m'
CBEIGEBG = '\33[46m'
CWHITEBG = '\33[47m'

CGREY = '\33[90m'
CRED2 = '\33[91m'
CGREEN2 = '\33[92m'
CYELLOW2 = '\33[93m'
CBLUE2 = '\33[94m'
CVIOLET2 = '\33[95m'
CBEIGE2 = '\33[96m'
CWHITE2 = '\33[97m'

CGREYBG = '\33[100m'
CREDBG2 = '\33[101m'
CGREENBG2 = '\33[102m'
CYELLOWBG2 = '\33[103m'
CBLUEBG2 = '\33[104m'
CVIOLETBG2 = '\33[105m'
CBEIGEBG2 = '\33[106m'
CWHITEBG2 = '\33[107m'


class Notification(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Display a custom message to the user and save it to the logs if required
    """

    def __init__(self, notif_type: int, message: str, log: bool=True) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check what type of notification has to be displayed

        PARAMETERS:
        -----------

        :param type->int : Index of the notification type flag
        :param message->str : Message to display

        RETURN:
        -------

        :return : result if input required, else None
        """
        self.log = log                      # Whether or not notifications should be written to logs
        self.response = ""                      # Allocated by self.__input(), returned by self.get()
        if notif_type == DEEP_NOTIF_INFO:
            self.__info(message)
        elif notif_type == DEEP_NOTIF_DEBUG:
            self.__debug(message)
        elif notif_type == DEEP_NOTIF_SUCCESS:
            self.__success(message)
        elif notif_type == DEEP_NOTIF_WARNING:
            self.__warning(message)
        elif notif_type == DEEP_NOTIF_ERROR:
            self.__error(message)
        elif notif_type == DEEP_NOTIF_FATAL:
            self.__fatal_error(message)
        elif notif_type == DEEP_NOTIF_INPUT:
            self.__input(message)
        elif notif_type == DEEP_NOTIF_RESULT:
            self.__result(message)
        else:
            raise ValueError("Unknown notification type: %s" % notif_type)

    def __fatal_error(self, message: str) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Display a FATAL ERROR message in RED BACKGROUND

        PARAMETERS:
        -----------

        :param message->str: The message to display

        RETURN:
        -------

        :return: None

        OTHERS:
        -------

        Close Deeplodocus Brain
        """
        message1 = "DEEP FATAL ERROR : %s" % message
        # message2 = "DEEP FATAL ERROR : Exiting the program"
        print("%s%s%s" % (CREDBG2, message1, CEND))
        # print("%s%s%s" % (CREDBG2, message2, CEND))
        if self.log is True:
            self.__add_log(message1)
            # self.__add_log(message2)
        # End(error=True)
        raise DeepError

    def __error(self, message: str) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Display an ERROR message in RED

        PARAMETERS:
        -----------

        :param message->str: The message to display

        RETURN:
        -------

        :return: None
        """
        message = "DEEP ERROR : %s" % message
        print("%s%s%s" % (CRED, message, CEND))
        if self.log is True:
            self.__add_log(message)

    def __warning(self, message: str)->None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Display an WARNING message in ORANGE/YELLOW

        PARAMETERS:
        -----------

        :param message->str: The message to display

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

        DESCRIPTION:
        ------------

        Display an DEBUG message in BEIGE

        PARAMETERS:
        -----------

        :param message->str: The message to display

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

        DESCRIPTION:
        ------------

        Display an SUCCESS message in GREEN

        PARAMETERS:
        -----------

        :param message->str: The message to display

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

        DESCRIPTION:
        ------------

        Display an INFO message in BLUE

        PARAMETERS:
        -----------

        :param message->str: The message to display

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
        Author: Alix Leroy, SW
        Displays a DEEP RESULT message in WHITE
        :param message: str: text to be printed
        :return: None
        """
        message = "DEEP RESULT : %s" % message
        print(message)
        if self.log is True:
            self.__add_log(message)

    def __input(self, message: str) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Display a BLINKING WHITE message and await for an input

        PARAMETERS:
        -----------

        :param message->str: The message to display

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

    def get(self) -> str:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the result of the input

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return self.response->str: The response givne by the user
        """
        return self.response

    @staticmethod
    def __add_log(message: str) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Add a message to the logs

        PARAMETERS:
        -----------

        :param message->str: The message to save in the logs

        RETURN:
        -------

        :return: None
        """
        Logs("notification", DEEP_PATH_NOTIFICATION, DEEP_EXT_LOGS).add(message)
