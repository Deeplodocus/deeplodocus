from deeplodocus.utils.notification import Notification
from deeplodocus.utils.flags.notif import *


class End(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Terminates the program
    """

    def __init__(self, error):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Terminates the program

        PARAMETERS:
        -----------

        :param error: Whether the program terminated with an error or not

        RETURN:
        -------

        :return: None
        """
        if error is False :
            self.__thanks_master()
        # Stop the program
        raise SystemExit(0)

    def __thanks_master(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Display thanks message

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: Universal Love <3
        """

        Notification(DEEP_NOTIF_INFO, "=================================")
        Notification(DEEP_NOTIF_INFO, "Thank you for using Deeplodocus !")
        Notification(DEEP_NOTIF_INFO, "== Made by Humans with deep <3 ==")
        Notification(DEEP_NOTIF_INFO, "=================================")
