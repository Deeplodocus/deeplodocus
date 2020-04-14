import torch

from deeplodocus.core.metrics.loss import Loss
from deeplodocus.core.metrics.metric import Metric
from deeplodocus.flags import *
from deeplodocus.flags.flag_lists import DEEP_LIST_DATASET
from deeplodocus.utils.deep_error import DeepError
from deeplodocus.utils.notification import Notification
from deeplodocus.utils.generic_utils import get_module, get_corresponding_flag


UNDERLINE = 50


class Metrics(object):

    def __init__(self, metrics=None):
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

    def reset(self, flag=None):
        flag = self.__check_flag(flag)
        if flag is None:
            self.values = {flag.name.lower(): {} for flag in DEEP_LIST_DATASET}
        else:
            self.values[flag.name.lower()] = {}

    def forward(self, flag, outputs, labels, inputs=None, additional_data=None):
        flag = self.__check_flag(flag)
        metrics = {}
        for metric_name in self.names:
            metrics[metric_name] = self.__dict__[metric_name].forward(outputs, labels, inputs, additional_data).item()
        self.__update_values(self.values[flag.name.lower()], metrics)
        return metrics

    def reduce(self, flag):
        flag = self.__check_flag(flag)
        return {
            metric_name: self.__dict__[metric_name].reduce_method(values)
            for metric_name, values in self.values[flag.name.lower()].items()
        }

    def summary(self):
        Notification(DEEP_NOTIF_INFO, "=" * UNDERLINE)
        Notification(DEEP_NOTIF_INFO, "SUMMARY OF METRICS :")
        Notification(DEEP_NOTIF_INFO, "-" * UNDERLINE)
        for name in self.names:
            Notification(DEEP_NOTIF_INFO, "%s :" % name)
            self.__dict__[name].summary(level=1)
        if not vars(self):
            Notification(DEEP_NOTIF_INFO, "None")
        Notification(DEEP_NOTIF_INFO, "=" * UNDERLINE)

    @staticmethod
    def __update_values(parent, new_values):
        for key, value in new_values.items():
            try:
                parent[key].append(value)
            except KeyError:
                parent[key] = [value]

    @staticmethod
    def __check_flag(flag):
        if flag is not None and not isinstance(flag, Flag):
            flag = get_corresponding_flag(flag, DEEP_LIST_DATASET)
        return flag


class Losses(object):

    def __init__(self, losses):
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
                if metric.module is None
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
        flag = self.__check_flag(flag)
        losses = {}
        for loss_name in self.names:
            losses[loss_name] = self.__dict__[loss_name].forward(outputs, labels, inputs, additional_data)
        self.__update_values(self.values[flag.name.lower()], losses)
        loss = sum([value for _, value in losses.items()])
        losses = {loss_name: value.item() for loss_name, value in losses.items()}
        return loss, losses

    def reset(self, flag=None):
        flag = self.__check_flag(flag)
        if flag is None:
            self.values = {flag.name.lower(): {} for flag in DEEP_LIST_DATASET}
        else:
            self.values[flag.name.lower()] = {}

    def reduce(self, flag):
        flag = self.__check_flag(flag)
        losses = {
            loss_name: sum(values) / len(values)
            for loss_name, values in self.values[flag.name.lower()].items()
        }
        loss = sum([value for _, value in losses.items()])
        losses = {loss_name: value.item() for loss_name, value in losses.items()}
        return loss, losses

    def summary(self):
        Notification(DEEP_NOTIF_INFO, "=" * UNDERLINE)
        Notification(DEEP_NOTIF_INFO, "SUMMARY OF LOSSES :")
        Notification(DEEP_NOTIF_INFO, "-" * UNDERLINE)
        for name in self.names:
            Notification(DEEP_NOTIF_INFO, "%s :" % name)
            self.__dict__[name].summary(level=1)
        if not vars(self):
            Notification(DEEP_NOTIF_INFO, "None")
        Notification(DEEP_NOTIF_INFO, "=" * UNDERLINE)

    @staticmethod
    def __check_flag(flag):
        if flag is not None and not isinstance(flag, Flag):
            flag = get_corresponding_flag(DEEP_LIST_DATASET, flag)
        return flag

    @staticmethod
    def __update_values(parent, new_values):
        for key, value in new_values.items():
            try:
                parent[key].append(value)
            except KeyError:
                parent[key] = [value]
