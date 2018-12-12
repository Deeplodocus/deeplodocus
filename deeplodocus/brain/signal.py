from deeplodocus.utils.flags.event import *


class Signal(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy


    DESCRIPTION:
    ------------

    Signal class to be used to interact with the Thalamus
    """

    def __init__(self, event, args={}):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Create a new signal

        PARAMETERS:
        -----------
        :param event:
        :param args:
        """
        self.event = event
        self.arguments = args

    def get_event(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Getter for event

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: event attribute
        """
        return self.event

    def get_arguments(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Getter for event

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: arguments attribute
        """
        return self.arguments