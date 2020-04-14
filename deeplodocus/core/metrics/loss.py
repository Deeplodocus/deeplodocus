import inspect
from typing import Union
from torch import nn

# Import Deeplodocus modules
from deeplodocus.flags import *
from deeplodocus.utils.deep_error import DeepError
from deeplodocus.utils.generic_utils import get_corresponding_flag, get_module
from deeplodocus.utils.notification import Notification


class Loss(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Loss class which stores, analyses a loss function

    """

    def __init__(
            self,
            name: str,
            module_path: Union[str, None],
            weight: float,
            kwargs: dict = None
    ):
        """
        AUTHORS:
        --------

        Samuel Westlake, Alix Leroy

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
        # Get the metric object
        method, module_path = get_module(
            name=name,
            module=module_path,
            browse={**DEEP_MODULE_LOSSES},
            silence=True
        )

        # If metric is not found by get_module, raise DEEP_FATAL
        if method is None:
            raise DeepError(DEEP_MSG_METRIC_NOT_FOUND % name)

        # If method is a class, initialise it with metric.kwargs
        kwargs = {} if kwargs is None else kwargs
        if inspect.isclass(method):
            method = method(**kwargs)

        self.name = name
        self.module_path = module_path
        self.method = method
        self.weight = weight
        self.is_custom = self.__check_custom()
        self.args = self.__check_args()
        self.kwargs = kwargs

    def forward(self, outputs, labels, inputs, additional_data):
        data = {
            DEEP_ENTRY_OUTPUT: outputs,
            DEEP_ENTRY_LABEL: labels,
            DEEP_ENTRY_INPUT: inputs,
            DEEP_ENTRY_ADDITIONAL_DATA: additional_data
        }
        args = {arg_name: data[entry_flag] for arg_name, entry_flag in self.args.items()}
        if inspect.isfunction(self.method):
            return self.method(*args, **self.kwargs) * self.weight
        else:
            return self.method.forward(**args) * self.weight

    def summary(self, level=0):
        Notification(DEEP_NOTIF_INFO, "%sname: %s" % ("  " * level, self.name))
        Notification(DEEP_NOTIF_INFO, "%smodule path: %s" % ("  " * level, self.module_path))
        if bool(self.args):
            Notification(DEEP_NOTIF_INFO, "%sargs: " % "  " * level)
            for arg_name, entry_flag in self.args.items():
                Notification(DEEP_NOTIF_INFO, "%s%s (%s)" % ("  " * (level + 1), arg_name, entry_flag.name))
        else:
            Notification(DEEP_NOTIF_INFO, "%sargs: []" % "  " * level, )
        if bool(self.kwargs):
            Notification(DEEP_NOTIF_INFO, "%skwargs: " % "  " * level)
            for kwarg_name, kwarg_value in self.kwargs.items():
                Notification(DEEP_NOTIF_INFO, "%s%s: %s" % ("  " * (level + 1), kwarg_name, kwarg_value))
        else:
            Notification(DEEP_NOTIF_INFO, "%skwargs: {}" % ("  " * level))
        Notification(DEEP_NOTIF_INFO, "%sweight: %s" % ("  " * level, self.f2str(self.weight)))

    def __check_args(self) -> dict:
        """
        AUTHORS:
        --------
        :author: Samuel Westlake, Alix Leroy

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
        if inspect.isfunction(self.method):
            args = inspect.getfullargspec(self.method).args
        else:
            args = inspect.getfullargspec(self.method.forward).args
            args.remove("self")
        args_dict = {}
        for arg in args:
            if self.is_custom:
                entry_flag = get_corresponding_flag(DEEP_LIST_ENTRY, arg, fatal=False)
                if entry_flag is None:
                    Notification(DEEP_NOTIF_FATAL, DEEP_MSG_LOSS_UNEXPECTED_ARG % arg)
                else:
                    args_dict[entry_flag] = arg
            else:
                if arg in ("input", "x", "inputs"):
                    args_dict[DEEP_ENTRY_OUTPUT] = arg
                elif arg in ("y", "y_hat", "label", "labels", "target", "targets"):
                    args_dict[DEEP_ENTRY_LABEL] = arg
                else:
                    Notification(DEEP_NOTIF_FATAL, DEEP_MSG_LOSS_UNEXPECTED_ARG % arg)
        self.__check_essential_args(args_dict)
        return {arg_name: entry_flag for entry_flag, arg_name in args_dict.items()}

    @staticmethod
    def __check_essential_args(args_dict):
        for entry_flag in (DEEP_ENTRY_OUTPUT, DEEP_ENTRY_LABEL):
            if entry_flag not in args_dict.keys():
                Notification(DEEP_NOTIF_WARNING, DEEP_MSG_METRIC_ARG_MISSING % entry_flag.name)

    def __check_custom(self) -> bool:
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
        main_module = self.method.__module__.split(".")[0]
        if main_module in DEEP_BACKEND_ALL:
            return False
        else:
            return True


    @staticmethod
    def f2str(x):
        x = str(x).rstrip("0")
        x = x + "0" if x.endswith(".") else x
        return x
