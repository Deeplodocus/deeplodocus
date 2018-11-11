from deeplodocus.utils.logs import Logs
from deeplodocus.utils.end import End
from deeplodocus.utils.flags import *
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
    Authors : Alix Leroy
    Display a custom message to the user
    """

    def __init__(self, type:int, message:str, write_logs:bool=True)->None:
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

        self.write_logs = write_logs

        # allocated, only used when requestion a DEEP_NOTIF_INPUT
        self.response = ""

        # Make sure the type is an integer
        if not isinstance(type, int):
            type = -1                   # DEEP_NOTIF_INFO by default

        # Make sure the message is a string
        message = str(message)

        # Send the message to the corresponding displayer
        if type == DEEP_NOTIF_INFO:
            self.__info(message)

        elif type == DEEP_NOTIF_DEBUG:
            self.__debug(message)

        elif type == DEEP_NOTIF_SUCCESS:
            self.__success(message)

        elif type == DEEP_NOTIF_WARNING:
            self.__warning(message)

        elif type == DEEP_NOTIF_ERROR:
            self.__error(message)

        elif type == DEEP_NOTIF_FATAL:
            self.__fatal_error(message)

        elif type == DEEP_NOTIF_INPUT:
            self.__input(message)

        # DEEP_NOTIF_INFO as default
        else:
            self.__info(message)



    def __fatal_error(self, message:str)->None:
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

        message1 = "DEEP FATAL ERROR : " + str(message)
        message2 = "DEEP FATAL ERROR : Exiting the program"

        print(CREDBG2 + str(message1) + CEND)
        print(CREDBG2 + str(message2) + CEND)

        if self.write_logs is True :
            self.__add_log(message1)
            self.__add_log(message2)

        End(error = True)

    def __error(self, message:str)->None:
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
        message = "DEEP ERROR : " + str(message)
        print(CRED + str(message) + CEND)

        if self.write_logs is True :
            # Add the the message to the log
            self.__add_log(message)

    def __warning(self, message:str)->None:
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
        message = "DEEP WARNING : " + str(message)
        print(CYELLOW2 + str(message) + CEND)

        if self.write_logs is True :
            # Add the the message to the log
            self.__add_log(message)

    def __debug(self, message:str)->None:
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

        message = "DEEP DEBUG : " + str(message)
        print(CBEIGE + str(message) + CEND)

        if self.write_logs is True :
            # Add the the message to the log
            self.__add_log(message)

    def __success(self, message:str)->None:
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
        message = "DEEP SUCCESS : " +str(message)
        print(CGREEN + str(message) + CEND)

        if self.write_logs is True :
            # Add the the message to the log
            self.__add_log(message)


    def __info(self, message:str)->None:
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

        message = "DEEP INFO : " + str(message)
        print(CBLUE + str(message) + CEND)

        if self.write_logs is True :
            # Add the the message to the log
            self.__add_log(message)


    def __input(self, message:str)->None:
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

        if self.write_logs is True :
            # Add the the message to the log
            self.__add_log(message)
            self.__add_log(str(self.response))


    def get(self)->str:
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


    def __add_log(self, message:str)->None:
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
        l = Logs("notification")
        l.add(message)



