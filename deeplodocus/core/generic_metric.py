import inspect
from typing import Union
from torch.nn import Module

from deeplodocus.utils.flags import *
from deeplodocus.utils.notification import Notification

Num = Union[int, float]

class GenericMetric(object):

    def __init__(self, name: str, method:Union[callable, Module], write_logs: bool = True):

        self.name = name
        self.write_logs = write_logs
        self.method = method
        self.arguments = []

    def get_name(self) -> str:
        return self.name

    def get_method(self) -> callable:
        return self.method

    def get_arguments(self) -> list:
        return self.arguments


    @staticmethod
    def __check_method(method) -> callable:
        return method

    @staticmethod
    def is_loss():
        return False