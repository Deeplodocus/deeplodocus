from deeplodocus.utils.notification import Notification
from deeplodocus.utils.flags import *


class Logo(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Display the logo of Deeplodocus containing the version number.
    """

    def __init__(self, version: str, write_logs: bool=True):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize the logo instance

        PARAMETERS:
        -----------

        :param version->str: Version of Deeplodocus

        RETURN:
        -------

        :return: None
        """

        self.__display(version, write_logs)

    @staticmethod
    def __display(version: str, write_logs: bool):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Display the full logo of Deeplodocus with corresponding colors and version number

        PARAMETERS:
        -----------

        :param version->str: Version of Deeplodocus

        RETURN:
        -------
        :return: None
        """

        Notification(DEEP_NOTIF_SUCCESS, "         `.-.````                                                                   ", write_logs=write_logs)
        Notification(DEEP_NOTIF_SUCCESS, "      ....0.0-....`                                                                 ", write_logs=write_logs)
        Notification(DEEP_NOTIF_SUCCESS, "      .:----::::s:-.`                                                               ", write_logs=write_logs)
        Notification(DEEP_NOTIF_SUCCESS, "        `..-::::/:---`                                                              ", write_logs=write_logs)
        Notification(DEEP_NOTIF_SUCCESS, "             .:/:----.`                                                             ", write_logs=write_logs)
        Notification(DEEP_NOTIF_SUCCESS, "              ://:----`                                                             ", write_logs=write_logs)
        Notification(DEEP_NOTIF_SUCCESS, "              :://----`                                                             ", write_logs=write_logs)
        Notification(DEEP_NOTIF_SUCCESS, "             `////----`                                                             ", write_logs=write_logs)
        Notification(DEEP_NOTIF_SUCCESS, "             ./:/:---..             `````````````                                   ", write_logs=write_logs)
        Notification(DEEP_NOTIF_SUCCESS, "             -/:/:---..          ``................``                               ", write_logs=write_logs)
        Notification(DEEP_NOTIF_SUCCESS, "             -://-----.        `---....------:------...````             `````````   ", write_logs=write_logs)
        Notification(DEEP_NOTIF_SUCCESS, "             -:/:-----.      `-:::-----------------::---......```..``.``......`     ", write_logs=write_logs)
        Notification(DEEP_NOTIF_SUCCESS, "             .:/:-.---.`````..---:---------------------:::---------..--.----.`      ", write_logs=write_logs)
        Notification(DEEP_NOTIF_INFO, "``'-.,_)`'-.,_)`'-.,_)`'-.,_)`'-.,_)`'-.,_)`'-.,_)`'-.,_)`'-.,_)`'-.,_)`'-.,_)`'-.,_)`'", write_logs=write_logs)
        Notification(DEEP_NOTIF_INFO, "``'-.,_,.-'``'-.,_,.='``'-.,_,.-'``'-.,_,.='````'-.,_,.-'``'-.,_,.='``'-.,_,.-'``'-.,_,", write_logs=write_logs)
        Notification(DEEP_NOTIF_INFO, "``'-.,_,.-'``'-.,_,.='``'-.,_,.-'``'-.,_,.='````'-.,_,.-'``'-.,_,.='``'-.,_,.-'``'-.,_,", write_logs=write_logs)
        Notification(DEEP_NOTIF_INFO, ".#####...######..######..#####...##.......####...#####....####....####...##..##...####..", write_logs=write_logs)
        Notification(DEEP_NOTIF_INFO, ".##..##..##......##......##..##..##......##..##..##..##..##..##..##..##..##..##..##.....", write_logs=write_logs)
        Notification(DEEP_NOTIF_INFO, ".##..##..####....####....#####...##......##..##..##..##..##..##..##......##..##...####..", write_logs=write_logs)
        Notification(DEEP_NOTIF_INFO, ".##..##..##......##......##......##......##..##..##..##..##..##..##..##..##..##......##.", write_logs=write_logs)
        Notification(DEEP_NOTIF_INFO, ".#####...######..######..##......######...####...#####....####....####....####....####..", write_logs=write_logs)
        Notification(DEEP_NOTIF_INFO, "........................................................................................", write_logs=write_logs)
        Notification(DEEP_NOTIF_INFO, "....................THE FRAMEWORK KEEPING YOUR HEAD ABOVE WATER.........................", write_logs=write_logs)
        Notification(DEEP_NOTIF_INFO, "..................................VERSION : " + str(version) + ".......................................", write_logs=write_logs)
        Notification(DEEP_NOTIF_INFO, "``'-.,_,.-'``'-.,_,.='``'-.,_,.-'``'-.,_,.='````'-.,_,.-'``'-.,_,.='``'-.,_,.-'``'-.,_,", write_logs=write_logs)
        print("\n")
