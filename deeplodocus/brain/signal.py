# Python imports
from typing import Optional

# Deeplodocus imports
from deeplodocus.utils.flag import Flag


class Signal(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy


    DESCRIPTION:
    ------------

    Signal class to be used to interact with the Thalamus
    """

    def __init__(self, event: Flag, args: Optional[dict] = None) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Create a new signal

        PARAMETERS:
        -----------
        :param event(Flag): The event the signal belongs to
        :param args(dict): The arguments to send with the signal
        """
        args = {} if args is None else args
        self.event = event
        self.arguments = args

    def get_event(self) -> Flag:
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

    def get_arguments(self) -> dict:
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

        :return: arguments attribute (dict)
        """
        return self.arguments
