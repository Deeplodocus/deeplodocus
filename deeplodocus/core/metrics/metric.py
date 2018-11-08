import inspect
from typing import Union
from torch.nn import Module

from deeplodocus.utils.flags import *
from deeplodocus.utils.notification import Notification
from deeplodocus.core.metrics.generic_metric import GenericMetric

class Metric(GenericMetric):

    def __init__(self, name:str, method:Union[callable, Module], write_logs:bool = True):
        super().__init__(name=name, method=method, write_logs=write_logs)
        self.method = self.__check_method(method)
        self.arguments = self.__check_arguments(method)

    @staticmethod
    def __check_method(method)->callable:
        if isinstance(method, Module):
            return method.forward
        else:
            return method

    @staticmethod
    def is_loss():
        return False

    def __check_arguments(self, method)->list:

        arguments = []

        if isinstance(method, Module):
            arguments_list =  inspect.getargspec(method.forward)[0]
        else:
            arguments_list = inspect.getargspec(method)[0]

        input_list= ["input", "x", "inputs"]
        output_list = ["out", "y_pred", "y_predicted", "output", "outputs"]
        label_list = ["y", "y_expect", "y_expected", "label", "labels", "target", "targets"]
        additional_data_list = ["additional_data", "aditional_data"]

        for arg in arguments_list:
            if arg in input_list:
                if isinstance(method, Module):
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
                Notification(DEEP_NOTIF_FATAL, "The following argument is not handled by the Deeplodocus metric system, please check the documentation : " + str(arg), write_logs=self.write_logs)
        return arguments