from deeplodocus.utils.notification import Notification
from deeplodocus.utils.flags import *
from deeplodocus import __version__

class Logo(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Display the logo of Deeplodocus containing the version number.
    """


    def __init__(self, version:str):
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

        version = __version__

        self.__display(version)

    def __display(self, version:str):
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


        Notification(DEEP_NOTIF_SUCCESS, "         `.-.````                                                                   ")
        Notification(DEEP_NOTIF_SUCCESS, "      ....0.0-....`                                                                 ")
        Notification(DEEP_NOTIF_SUCCESS, "      .:----::::s:-.`                                                               ")
        Notification(DEEP_NOTIF_SUCCESS, "        `..-::::/:---`                                                              ")
        Notification(DEEP_NOTIF_SUCCESS, "             .:/:----.`                                                             ")
        Notification(DEEP_NOTIF_SUCCESS, "              ://:----`                                                             ")
        Notification(DEEP_NOTIF_SUCCESS, "              :://----`                                                             ")
        Notification(DEEP_NOTIF_SUCCESS, "             `////----`                                                             ")
        Notification(DEEP_NOTIF_SUCCESS, "             ./:/:---..             `````````````                                   ")
        Notification(DEEP_NOTIF_SUCCESS, "             -/:/:---..          ``................``                               ")
        Notification(DEEP_NOTIF_SUCCESS, "             -://-----.        `---....------:------...````             `````````   ")
        Notification(DEEP_NOTIF_SUCCESS, "             -:/:-----.      `-:::-----------------::---......```..``.``......`     ")
        Notification(DEEP_NOTIF_SUCCESS, "             .:/:-.---.`````..---:---------------------:::---------..--.----.`      ")
        Notification(DEEP_NOTIF_INFO, "``'-.,_)`'-.,_)`'-.,_)`'-.,_)`'-.,_)`'-.,_)`'-.,_)`'-.,_)`'-.,_)`'-.,_)`'-.,_)`'-.,_)`'")
        Notification(DEEP_NOTIF_INFO, "``'-.,_,.-'``'-.,_,.='``'-.,_,.-'``'-.,_,.='````'-.,_,.-'``'-.,_,.='``'-.,_,.-'``'-.,_,")
        Notification(DEEP_NOTIF_INFO, "``'-.,_,.-'``'-.,_,.='``'-.,_,.-'``'-.,_,.='````'-.,_,.-'``'-.,_,.='``'-.,_,.-'``'-.,_,")
        Notification(DEEP_NOTIF_INFO, ".#####...######..######..#####...##.......####...#####....####....####...##..##...####..")
        Notification(DEEP_NOTIF_INFO, ".##..##..##......##......##..##..##......##..##..##..##..##..##..##..##..##..##..##.....")
        Notification(DEEP_NOTIF_INFO, ".##..##..####....####....#####...##......##..##..##..##..##..##..##......##..##...####..")
        Notification(DEEP_NOTIF_INFO, ".##..##..##......##......##......##......##..##..##..##..##..##..##..##..##..##......##.")
        Notification(DEEP_NOTIF_INFO, ".#####...######..######..##......######...####...#####....####....####....####....####..")
        Notification(DEEP_NOTIF_INFO, "........................................................................................")
        Notification(DEEP_NOTIF_INFO, "...................THE FRAMEWORK THAT KEEPS YOUR HEAD ABOVE WATER.......................")
        Notification(DEEP_NOTIF_INFO, "..................................VERSION : " + str(version) + ".......................................")
        Notification(DEEP_NOTIF_INFO, "``'-.,_,.-'``'-.,_,.='``'-.,_,.-'``'-.,_,.='````'-.,_,.-'``'-.,_,.='``'-.,_,.-'``'-.,_,")
        print("\n")



