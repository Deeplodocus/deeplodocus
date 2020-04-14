import inspect
from typing import Union

from deeplodocus.flags import *
from deeplodocus.utils.deep_error import DeepError
from deeplodocus.utils.generic_utils import get_corresponding_flag, get_module
from deeplodocus.utils.notification import Notification


class Metric(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy
    :author: Samuel Westlake

    DESCRIPTION:
    ------------

    A class which contains any metric to be computed

    """
    def __init__(
            self, name: str,
            module_path: Union[str, None],
            reduce: str = "mean",
            kwargs: dict = None
    ):
        """
        AUTHORS:
        --------

        Samuel Westlake

        DESCRIPTION:
        ------------

        Initialize a Metric instance

        PARAMETERS:
        -----------

        :param name->str: The name of the metric
        :param method->Union[callable, torch.nn.Module]: The method to be computed

        RETURN:
        -------

        :return: None
        """
        # Get the metric object
        method, module_path = get_module(
            name=name,
            module=module_path,
            browse={**DEEP_MODULE_METRICS, **DEEP_MODULE_LOSSES},
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
        self.args = self.__check_args()
        self.kwargs = kwargs
        self.reduce_method = None
        self._reduce = get_corresponding_flag(DEEP_LIST_REDUCE, reduce)
        self.__set_reduce_method()

    def forward(self, outputs, labels, inputs, additional_data):
        data = {
            DEEP_ENTRY_OUTPUT: outputs,
            DEEP_ENTRY_LABEL: labels,
            DEEP_ENTRY_INPUT: inputs,
            DEEP_ENTRY_ADDITIONAL_DATA: additional_data
        }
        args = {arg_name: data[entry_flag] for arg_name, entry_flag in self.args.items()}
        if inspect.isfunction(self.method):
            return self.method(**args, **self.kwargs)
        else:
            return self.method.forward(**args)

    def __set_reduce_method(self):
        """
        AUTHORS:
        --------

        Samuel Westlake

        DESCRIPTION:
        ------------

        Set self.reduce_method to the method to use when reducing a list of metric values

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        if self._reduce.corresponds(DEEP_REDUCE_MEAN):
            self.reduce_method = mean
        elif self._reduce.corresponds(DEEP_REDUCE_SUM):
            self.reduce_method = sum
        elif self._reduce.corresponds(DEEP_REDUCE_LAST):
            return last

    def __check_args(self):
        if inspect.isfunction(self.method):
            args = inspect.getfullargspec(self.method).args
        else:
            args = inspect.getfullargspec(self.method.forward).args
            args.remove("self")
        args_dict = {}
        for arg_name in args:
            entry_flag = get_corresponding_flag(DEEP_LIST_ENTRY, arg_name, fatal=False)
            if entry_flag is None:
                Notification(DEEP_NOTIF_FATAL, DEEP_MSG_METRIC_UNEXPECTED_ARG % arg_name)
            else:
                args_dict[entry_flag] = arg_name
        self.__check_essential_args(args_dict)
        return {arg_name: entry_flag for entry_flag, arg_name in args_dict.items()}

    @staticmethod
    def __check_essential_args(args_dict):
        for entry_flag in (DEEP_ENTRY_OUTPUT, DEEP_ENTRY_LABEL):
            if entry_flag not in args_dict.keys():
                Notification(DEEP_NOTIF_WARNING, DEEP_MSG_METRIC_ARG_MISSING % entry_flag.name)

    @property
    def reduce(self):
        return self._reduce

    @reduce.setter
    def reduce(self, new_value):
        """
        AUTHORS:
        --------

        Samuel Westlake

        DESCRIPTION:
        ------------

        Handles automatic updating of self.reduce_method when self.reduce is changed.

        PARAMETERS:
        -----------

        new_value: str or Flag: Which method to use whe reducing list of metric values ("mean", "sum", "last")

        RETURN:
        -------

        :return: None
        """
        self._reduce = get_corresponding_flag(DEEP_LIST_REDUCE, new_value)
        self.__set_reduce_method()

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
        Notification(DEEP_NOTIF_INFO, "%sreduce_method: %s" % ("  " * level, self.reduce.name))


def mean(x):
    return sum(x) / len(x)


def last(x):
    return x[-1]
