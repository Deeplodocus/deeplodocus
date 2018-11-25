from typing import Union
from torch.nn import Module

Num = Union[int, float]

class GenericMetric(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Generic class for Metric and Loss
    """

    def __init__(self, name: str, method:Union[callable, Module]):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize a GenericMetric

        PARAMETERS:
        -----------

        :param name->str:
        :param method->Union[callable, Module]: Either a method for a Metric or a torch.nn.Module for a Metric or a Loss

        RETURN:
        -------

        :return:None
        """
        self.name = name
        self.method = method
        self.arguments = []

    def get_name(self) -> str:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the name of the GenericMetric

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return->str: The name of the GenericMetric
        """
        return self.name

    def get_method(self) -> callable:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the method to be called by Deeplodocus

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return->callable: The method to be called by Deeplodocus
        """
        return self.method

    def get_arguments(self) -> list:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the list of arguments required to use within the method

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return->list: The list of arguments
        """
        return self.arguments


    @staticmethod
    def __check_method(method) -> callable:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check what type of method has ot be used

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return->callable: The method to be used by Deeplodocus
        """
        return method

    @staticmethod
    def is_loss():
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check if the GenericMetric uses a loss function or not

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return->bool: Whether the GenericMetric method is a loss function or not
        """
        return False