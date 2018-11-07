import inspect
from typing import Union
from torch.nn import Module

from deeplodocus.utils.flags import *
from deeplodocus.utils.notification import Notification
from deeplodocus.core.generic_metric import GenericMetric

Num = Union[int, float]

class Loss(GenericMetric):

    def __init__(self, name:str, loss:Module, is_custom=False, weight:Num=1.0, write_logs:bool = True):
        super().__init__(name=name, method=loss, write_logs=write_logs)
        self.is_custom = is_custom
        self.weight = weight
        self.arguments = self.__check_arguments(loss.forward)


    def get_weight(self)->Num:
        return self.weight

    @staticmethod
    def is_loss():
        return True

    def __check_arguments(self, loss)->list:

        arguments = []

        arguments_list =  inspect.getargspec(loss)[0]

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
                Notification(DEEP_NOTIF_FATAL, "The following argument is not handled by the Deeplodocus loss system, please check the documentation : " + str(arg), write_logs=self.write_logs)
        return arguments