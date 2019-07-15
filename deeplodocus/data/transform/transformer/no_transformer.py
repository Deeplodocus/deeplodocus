# Deeplodocus imports
from deeplodocus.utils.notification import Notification

# Flags imports
from deeplodocus.utils.flags.notif import *


class NoTransformer(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    NoTransformer class which doesn't transform any data
    """

    def __init__(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize an NoTransformer instance

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """

        pass

    @staticmethod
    def summary():
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Print the summary of the NoTransformer

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        Notification(DEEP_NOTIF_INFO, "------------------------------------")
        Notification(DEEP_NOTIF_INFO, "No Transformer for this entry")
        Notification(DEEP_NOTIF_INFO, "------------------------------------")

    @staticmethod
    def has_transforms():
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Whether the Transform has transforms or not

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        return False

    @staticmethod
    def reset():
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Required for compatibility with Transformers

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        pass
