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
        :return:
        """
        return self.event

    def get_arguments(self):
        return self.arguments