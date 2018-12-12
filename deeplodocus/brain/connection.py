import weakref
from typing import Union
from typing import List


class Connection(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Connection class allowing to gather information for a connection in the Thalamus.
    Contains a weak reference to a receiver method and the expected arguments
    """

    def __init__(self, receiver: callable, expected_arguments: Union[List, None] = None) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize the connection with a weak reference to the method to connect

        PARAMETERS:
        -----------

        :param receiver(callable): The method to connect
        :param expected_arguments(Union[List, None]): The list of expected arguments (None means all the arguments)

        RETURN:
        -------

        :return: None

        """
        # Get the weak reference of the method to call
        self.receiver = weakref.WeakMethod(receiver)
        self.expected_arguments = expected_arguments

    def get_receiver(self) -> callable:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Getter for self.receiver

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: self.receiver
        """
        return self.receiver

    def get_expected_arguments(self) -> Union[List, None]:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Getter for self.expected_arguments

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: self.expected_arguments
        """
        return self.expected_arguments
