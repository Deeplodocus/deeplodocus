from deeplodocus.utils.notification import Notification
from deeplodocus.utils.flags.notif import *


class Metrics(object):

    def __init__(self, metrics=None):
        if metrics is not None:
            self.__dict__ = metrics

    def add(self, metric):
        self.__dict__.update(metric)

    def summary(self):
        Notification(DEEP_NOTIF_INFO, '================================================================')
        Notification(DEEP_NOTIF_INFO, "LIST OF METRICS :")
        Notification(DEEP_NOTIF_INFO, '================================================================')
        for metric_name, metric in vars(self).items():
            Notification(DEEP_NOTIF_INFO, "%s : %s" % (metric_name, metric.method))
        if not vars(self):
            Notification(DEEP_NOTIF_INFO, "None")
        Notification(DEEP_NOTIF_INFO, "")

    def get(self, item):
        return self.__dict__[item]


class Losses(object):

    def __init__(self, losses=None):
        if losses is not None:
            self.__dict__ = losses

    def add(self, metric):
        self.__dict__.update(metric)

    def summary(self):
        Notification(DEEP_NOTIF_INFO, '================================================================')
        Notification(DEEP_NOTIF_INFO, "LIST OF LOSS FUNCTIONS :")
        Notification(DEEP_NOTIF_INFO, '================================================================')
        for loss_name, loss in vars(self).items():
            Notification(DEEP_NOTIF_INFO, "%s : %s" % (loss_name, loss.method))
        if not vars(self):
            Notification(DEEP_NOTIF_INFO, "None")
        Notification(DEEP_NOTIF_INFO, "")

    def get(self, item):
        return self.__dict__[item]