from deeplodocus.utils.logs import Logs
from deeplodocus.utils.end import End
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

#
# DEEPLODOCUS FLAGS
#


DEEP_INFO = 0
DEEP_DEBUG = 1
DEEP_SUCCESS = 2
DEEP_WARNING = 3
DEEP_ERROR = 4
DEEP_FATAL = 5
DEEP_INPUT = 6



class Notification(object):
    """
    Authors : Alix Leroy
    Display a custom message to the user
    """

    def __init__(self, type, message, write_logs=True):
        """
        Authors : Alix Leroy
        :param type: str : Type of message (fatal, error, warning, info, success)
        :param message: str : Message to display
        :return : result if input required, else None
        """

        self.write_logs = write_logs
        self.response = ""

        if not isinstance(type, int):
            type = -1                   # Info by default

        message = str(message)



        if type == 0:
            self.__info(message)

        elif type == 1:
            self.__debug(message)

        elif type == 2:
            self.__success(message)

        elif type == 3:
            self.__warning(message)

        elif type == 4:
            self.__error(message)

        elif type == 5:
            self.__fatal_error(message)

        elif type == 6:
            self.__input(message)

        # Info notification as default
        else:
            self.__info(message)



    def __fatal_error(self, message):
        """
        Authors : Alix Leroy
        Display a FATAL ERROR message in RED BACKGROUND
        :param message: the message to display
        :return: Exit the program
        """

        message1 = "DEEP FATAL ERROR : " + str(message)
        message2 = "DEEP FATAL ERROR : Exiting the program"

        print(CREDBG2 + str(message1) + CEND)
        print(CREDBG2 + str(message2) + CEND)

        if self.write_logs is True :
            self.__add_log(message1)
            self.__add_log(message2)

        End(error = True)

    def __error(self, message):
        """
        Authors : Alix Leroy
        Display an ERROR message in RED
        :param message: the message to display
        :return: None
        """
        message = "DEEP ERROR : " + str(message)
        print(CRED + str(message) + CEND)

        if self.write_logs is True :
            # Add the the message to the log
            self.__add_log(message)

    def __warning(self, message):
        """
        Authors : Alix Leroy
        Display an WARNING message in ORANGE/YELLOW
        :param message: the message to display
        :return: None
        """
        message = "DEEP WARNING : " + str(message)
        print(CYELLOW2 + str(message) + CEND)

        if self.write_logs is True :
            # Add the the message to the log
            self.__add_log(message)

    def __debug(self, message):
        """
        Authors : Alix Leroy
        Display an DEBUG message in BEIGE
        :param message: the message to display
        :return: None
        """
        message = "DEEP DEBUG : " + str(message)
        print(CBEIGE + str(message) + CEND)

        if self.write_logs is True :
            # Add the the message to the log
            self.__add_log(message)

    def __success(self, message):
        """
        Authors : Alix Leroy
        Display an SUCCESS message in GREEN
        :param message: the message to display
        :return: None
        """
        message = "DEEP SUCCESS : " +str(message)
        print(CGREEN + str(message) + CEND)

        if self.write_logs is True :
            # Add the the message to the log
            self.__add_log(message)


    def __info(self, message):
        """
        Authors : Alix Leroy
        Display an INFO message in BLUE
        :param message: the message to display
        :return: None
        """
        message = "DEEP INFO : " + str(message)
        print(CBLUE + str(message) + CEND)

        if self.write_logs is True :
            # Add the the message to the log
            self.__add_log(message)


    def __input(self, message):
        """
        Authors : Alix Leroy
        Display a message for an input
        :param message: the message to display
        :return:None
        """

        message = "DEEP INPUT : " + str(message)
        print(CBLINK + CBOLD + str(message) + CEND)

        self.response = input(">")


        if self.write_logs is True :
            # Add the the message to the log
            self.__add_log(message)
            self.__add_log(str(self.response))


    def get(self):
        """
        Authors : Alix Leroy
        Get the result of the input
        :return: The result of the input
        """
        return self.response


    def __add_log(self, message):
        """
        Authors : Alix Leroy,
        :param message:
        :return: None
        """

        l = Logs("notification")
        l.add(message)



