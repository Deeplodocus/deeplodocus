from deeplodocus.utils.notification import Notification
from deeplodocus.utils.types import *
from deeplodocus import __version__

class Logo(object):
    """
    Authors : Alix Leroy,
    Display the logo of deeplodocus
    """


    def __init__(self, version):
        """
        Authors : ALix Leroy,
        Initialize the logo class
        :param version: string of the version of deeeplodocus
        """

        version = __version__

        self.__display(version)

    def __display(self, version):
        """
        Authors : Alix Leroy,
        Display the logo of Deeplodocus
        :param version: string : version of deeplodocus
        :return: None
        """


        Notification(DEEP_SUCCESS, "         `.-.````                                                                   ")
        Notification(DEEP_SUCCESS, "      ....0.0-....`                                                                 ")
        Notification(DEEP_SUCCESS, "      .:----::::s:-.`                                                               ")
        Notification(DEEP_SUCCESS, "        `..-::::/:---`                                                              ")
        Notification(DEEP_SUCCESS, "             .:/:----.`                                                             ")
        Notification(DEEP_SUCCESS, "              ://:----`                                                             ")
        Notification(DEEP_SUCCESS, "              :://----`                                                             ")
        Notification(DEEP_SUCCESS, "             `////----`                                                             ")
        Notification(DEEP_SUCCESS, "             ./:/:---..             `````````````                                   ")
        Notification(DEEP_SUCCESS, "             -/:/:---..          ``................``                               ")
        Notification(DEEP_SUCCESS, "             -://-----.        `---....------:------...````             `````````   ")
        Notification(DEEP_SUCCESS, "             -:/:-----.      `-:::-----------------::---......```..``.``......`     ")
        Notification(DEEP_SUCCESS, "             .:/:-.---.`````..---:---------------------:::---------..--.----.`      ")
        Notification(DEEP_INFO, "``'-.,_)`'-.,_)`'-.,_)`'-.,_)`'-.,_)`'-.,_)`'-.,_)`'-.,_)`'-.,_)`'-.,_)`'-.,_)`'-.,_)`'")
        Notification(DEEP_INFO, "``'-.,_,.-'``'-.,_,.='``'-.,_,.-'``'-.,_,.='````'-.,_,.-'``'-.,_,.='``'-.,_,.-'``'-.,_,")
        Notification(DEEP_INFO, "``'-.,_,.-'``'-.,_,.='``'-.,_,.-'``'-.,_,.='````'-.,_,.-'``'-.,_,.='``'-.,_,.-'``'-.,_,")
        Notification(DEEP_INFO, ".#####...######..######..#####...##.......####...#####....####....####...##..##...####..")
        Notification(DEEP_INFO, ".##..##..##......##......##..##..##......##..##..##..##..##..##..##..##..##..##..##.....")
        Notification(DEEP_INFO, ".##..##..####....####....#####...##......##..##..##..##..##..##..##......##..##...####..")
        Notification(DEEP_INFO, ".##..##..##......##......##......##......##..##..##..##..##..##..##..##..##..##......##.")
        Notification(DEEP_INFO, ".#####...######..######..##......######...####...#####....####....####....####....####..")
        Notification(DEEP_INFO, "........................................................................................")
        Notification(DEEP_INFO, "...................THE FRAMEWORK THAT KEEPS YOUR HEAD ABOVE WATER.......................")
        Notification(DEEP_INFO, "..................................VERSION : " + str(version) + ".......................................")
        Notification(DEEP_INFO, "``'-.,_,.-'``'-.,_,.='``'-.,_,.-'``'-.,_,.='````'-.,_,.-'``'-.,_,.='``'-.,_,.-'``'-.,_,")
        print("\n")



