import inspect
from typing import Union

from deeplodocus.flags import *
from deeplodocus.flags.flag_lists import DEEP_LIST_DATASET
from deeplodocus.utils.deep_error import DeepError
from deeplodocus.utils.notification import Notification
from deeplodocus.utils.generic_utils import get_corresponding_flag, get_module


UNDERLINE = 50


class GenericMetrics(object):

    def __init__(self):
        self.values = {}
        self.names = []

    def reset(self, flag=None):
        flag = get_corresponding_flag(DEEP_LIST_DATASET, flag, fatal=False)
        if flag is None:
            self.values = {flag.name.lower(): {} for flag in DEEP_LIST_DATASET}
        else:
            self.values[flag.name.lower()] = {}

    def summary__(self, title="SUMMARY OF METRICS :"):
        Notification(DEEP_NOTIF_INFO, "=" * UNDERLINE)
        Notification(DEEP_NOTIF_INFO, title)
        Notification(DEEP_NOTIF_INFO, "-" * UNDERLINE)
        for name in self.names:
            Notification(DEEP_NOTIF_INFO, "%s :" % name)
            self.__dict__[name].summary(level=1)
        if not vars(self):
            Notification(DEEP_NOTIF_INFO, "None")
        Notification(DEEP_NOTIF_INFO, "=" * UNDERLINE)

    @staticmethod
    def update_values(parent, new_values):
        for key, value in new_values.items():
            try:
                parent[key].append(value)
            except KeyError:
                parent[key] = [value]


class Metrics(GenericMetrics):

    def __init__(self, metrics=None):
        super(Metrics, self).__init__()
        self.names = []
        if metrics is not None:
            for name, metric in metrics.items():
                self.add(name, metric)
        self.values = {flag.name.lower(): {} for flag in DEEP_LIST_DATASET}

    def add(self, name, metric):
        # Notify the user which metric is being collected and from where
        Notification(
            DEEP_NOTIF_INFO,
            DEEP_MSG_METRIC_LOADING % (
                "%s : %s from default modules" % (name, metric.name)
                if metric.module is None
                else "%s : %s from %s" % (name, metric.name, metric.module)
            )
        )
        try:
            # Try to initialise metric, add to self__dict__ and store its name
            m = Metric(
                name=metric.name,
                module_path=metric.module,
                reduce=metric.reduce,
                kwargs=metric.kwargs.get()
            )
            self.__dict__[name] = m
            self.names.append(name)
            Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_METRIC_LOADED % (name, m.name, m.module_path))
        except DeepError:
            Notification(DEEP_NOTIF_ERROR, DEEP_MSG_METRIC_NOT_FOUND % name)

    def forward(self, flag, outputs, labels, inputs=None, additional_data=None):
        flag = get_corresponding_flag(DEEP_LIST_DATASET, flag, fatal=False)
        metrics = {}
        for metric_name in self.names:
            metrics[metric_name] = self.__dict__[metric_name].forward(outputs, labels, inputs, additional_data).item()
        self.update_values(self.values[flag.name.lower()], metrics)
        return metrics

    def reduce(self, flag):
        flag = get_corresponding_flag(DEEP_LIST_DATASET, flag, fatal=False)
        return {
            metric_name: self.__dict__[metric_name].reduce_method(values)
            for metric_name, values in self.values[flag.name.lower()].items()
        }

    def summary(self):
        self.summary__()


class Losses(GenericMetrics):

    def __init__(self, losses):
        super(Losses, self).__init__()
        self.names = []
        for name, loss in losses.items():
            self.add(name, loss)
        self.values = {flag.name.lower(): {} for flag in DEEP_LIST_DATASET}

    def add(self, name, loss):
        # Notify the user which loss is being collected and from where
        Notification(
            DEEP_NOTIF_INFO,
            DEEP_MSG_LOSS_LOADING % (
                "%s : %s from default modules" % (name, loss.name)
                if loss.module is None
                else "%s : %s from %s" % (name, loss.name, loss.module)
            )
        )

        try:
            # Try to initialise metric, add to self__dict__ and store its name
            m = Loss(
                name=loss.name,
                module_path=loss.module,
                weight=loss.weight,
                kwargs=loss.kwargs.get()
            )
            self.__dict__[name] = m
            self.names.append(name)
            Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_LOSS_LOADED % (name, m.name, m.module_path))
        except DeepError:
            Notification(DEEP_NOTIF_ERROR, DEEP_MSG_LOSS_NOT_FOUND % name)

    def forward(self, flag, outputs, labels, inputs=None, additional_data=None):
        flag = get_corresponding_flag(DEEP_LIST_DATASET, flag, fatal=False)
        losses = {}
        for loss_name in self.names:
            losses[loss_name] = self.__dict__[loss_name].forward(outputs, labels, inputs, additional_data)
        self.update_values(self.values[flag.name.lower()], losses)
        loss = sum([value for _, value in losses.items()])
        losses = {loss_name: value.item() for loss_name, value in losses.items()}
        return loss, losses

    def reduce(self, flag):
        flag = get_corresponding_flag(DEEP_LIST_DATASET, flag, fatal=False)
        losses = {
            loss_name: sum(values) / len(values)
            for loss_name, values in self.values[flag.name.lower()].items()
        }
        loss = sum([value for _, value in losses.items()])
        losses = {loss_name: value.item() for loss_name, value in losses.items()}
        return loss, losses

    def summary(self):
        self.summary__(title="SUMMARY OF LOSSES :")


class Metric(object):

    def __init__(self, name: str, module_path: Union[str, None], reduce: str = "mean", kwargs: dict = None):
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

    def summary(self, level=0):
        Notification(DEEP_NOTIF_INFO, "%sname: %s" % ("  " * level, self.name))
        Notification(DEEP_NOTIF_INFO, "%smodule path: %s" % ("  " * level, self.module_path))
        Notification(DEEP_NOTIF_INFO, "%sreduce_method: %s" % ("  " * level, self.reduce.name))
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

    @property
    def reduce(self):
        return self._reduce

    @reduce.setter
    def reduce(self, new_value):
        self._reduce = get_corresponding_flag(DEEP_LIST_REDUCE, new_value)
        self.__set_reduce_method()

    def __set_reduce_method(self):
        if self._reduce.corresponds(DEEP_REDUCE_MEAN):
            self.reduce_method = self.mean
        elif self._reduce.corresponds(DEEP_REDUCE_SUM):
            self.reduce_method = sum
        elif self._reduce.corresponds(DEEP_REDUCE_LAST):
            return self.last

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

    @staticmethod
    def mean(x):
        return sum(x) / len(x)

    @staticmethod
    def last(x):
        return x[-1]


class Loss(object):

    def __init__(self, name: str, module_path: Union[str, None], weight: float, kwargs: dict = None):
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
        self.is_custom = self.check_custom()
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
        Notification(DEEP_NOTIF_INFO, "%sweight: %s" % ("  " * level, self.f2str(self.weight)))
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

    def __check_args(self) -> dict:
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
        self.check_essential_args(args_dict)
        return {arg_name: entry_flag for entry_flag, arg_name in args_dict.items()}

    @staticmethod
    def check_essential_args(args_dict):
        for entry_flag in (DEEP_ENTRY_OUTPUT, DEEP_ENTRY_LABEL):
            if entry_flag not in args_dict.keys():
                Notification(DEEP_NOTIF_WARNING, DEEP_MSG_METRIC_ARG_MISSING % entry_flag.name)

    def check_custom(self) -> bool:
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


class OverWatch(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Metric to overwatch during the training
    """

    def __init__(self, name: str = DEEP_LOG_TOTAL_LOSS, condition: Union[Flag, int, str, None] = DEEP_SAVE_CONDITION_LESS):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize the OverWatchMetric instance

        PARAMETERS:
        -----------
        :param name (str): The name of the metric to over watch
        :param condition (Flag):
        """
        self.name = name
        self.value = 0.0
        self.condition = get_corresponding_flag(
            flag_list=DEEP_LIST_SAVE_CONDITIONS,
            info=condition,
            fatal=False,
            default=DEEP_SAVE_CONDITION_LESS
        )

    def set_value(self, value: float):
        self.value = value

    def get_name(self):
        return self.name

    def get_value(self):
        return self.value

    def get_condition(self):
        return self.condition
