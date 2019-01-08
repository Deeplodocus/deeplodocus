# Import python modules
import inspect
from typing import Union

# Import back-end modules
import torch

# Import Deeplodocus modules
from deeplodocus.utils.flags.backend import DEEP_BACKEND_ALL
from deeplodocus.utils.flags.entry import *
from deeplodocus.utils.flags.notif import DEEP_NOTIF_FATAL
from deeplodocus.utils.notification import Notification
from deeplodocus.core.metrics.generic_metric import GenericMetric

Num = Union[int, float]


class Loss(GenericMetric):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Loss class which stores, analyses a loss function

    """

    def __init__(self, name: str, loss: torch.nn.Module, weight: Num=1.0):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize a Loss instance.
        Check if it is a custom loss
        Check the arguments the forward method contains

        PARAMETERS:
        -----------

        :param name(str): The name of the loss function
        :param loss(torch.nn.Module): The loss callable
        :param weight(float): The weight of the loss in the total loss function

        RETURN:
        -------

        :return: None
        """
        super().__init__(name=name, method=loss)
        self.is_custom = self.check_custom(loss)
        self.weight = weight
        self.arguments = self.__check_arguments(loss.forward) # loss.forward will be called automatically, but keep loss as function to call

    def check_custom(self, loss: torch.nn.Module):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check whether the loss function is a custom one or not

        PARAMETERS:
        -----------

        :param loss(torch.nn.Module):

        RETURN:
        -------

        :return (bool): Whether the loss function is a custom one or not
        """
        # Get the main module's name
        main_module = loss.__module__.split(".")[0]

        if main_module in DEEP_BACKEND_ALL:
            return False
        else:
            return True

    def get_weight(self)->Num:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the weight of the loss function

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return(Num): self.weight
        """
        return self.weight

    @staticmethod
    def is_loss():
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Return whether the generic metric is a loss or not

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return (bool): True
        """
        return True

    def __check_arguments(self, loss: callable)->list:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check the arguments required for the loss function

        PARAMETERS:
        -----------

        :param loss(callable): The method to analyze

        RETURN:
        -------

        :return arguments(list): The list of arguments DEEP flags
        """

        arguments = []

        arguments_list = inspect.getfullargspec(loss)[0]

        input_list= ["input", "x", "inputs"]
        output_list = ["out", "y_pred", "y_predicted", "output", "outputs"]
        label_list = ["y", "y_expect", "y_expected", "label", "labels", "target", "targets"]
        additional_data_list = ["additional_data", "aditional_data"]

        for arg in arguments_list:
            if arg in input_list:
                if self.is_custom is False:
                    arguments.append(DEEP_ENTRY_OUTPUT)
                else:
                    arguments.append(DEEP_ENTRY_INPUT)
            elif arg in output_list:
                arguments.append(DEEP_ENTRY_OUTPUT)
            elif arg in label_list:
                arguments.append(DEEP_ENTRY_LABEL)
            elif arg in additional_data_list:
                arguments.append(DEEP_ENTRY_ADDITIONAL_DATA)
            elif arg == "self":
                continue
            else:
                Notification(DEEP_NOTIF_FATAL, "The following argument is not handled by the Deeplodocus loss system, please check the documentation : " + str(arg))
        return arguments
