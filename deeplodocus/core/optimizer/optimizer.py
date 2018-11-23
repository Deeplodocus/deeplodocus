from deeplodocus.utils.notification import Notification
from deeplodocus.utils.flags.filter import *
from deeplodocus.utils.flags.notif import *
from deeplodocus.utils.dict_utils import check_kwargs
from deeplodocus.utils.generic_utils import get_module


class Optimizer(object):

    def __init__(self, params, config):
        self.optimizer = None
        self.params = params
        self.config = config
        self.load_optimizer()

    def load_optimizer(self):
        """
        :return:
        """
        optimizer = get_module(module=self.config.module,
                               name=self.__format_optimizer_name(self.config.name))
        kwargs = check_kwargs(self.config.kwargs)
        self.optimizer = optimizer(self.params, **kwargs)

    def get(self):
        """
        :return:
        """
        return self.optimizer

    @staticmethod
    def __format_optimizer_name(name: str):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Format the name of the optimizer

        PARAMETERS:
        -----------

        :param name->str: The name of the optimizer

        RETURN:
        -------

        :return name->str: The formatted name of the optimizer
        """

        if name.lower() in DEEP_FILTER_OPTIMIZERS:
            Notification(DEEP_NOTIF_FATAL, "The following optimizer is not allowed : %s" % name)
        if name.lower() == "sgd":
            name = "SGD"
        elif name.lower == "adam":
            name = "Adam"
        elif name.lower() == "adamax":
            name = "Adamax"
        elif name.lower == "asgd":
            name = "ASGD"
        elif name.lower() == "lbfgs":
            name = "LBFGS"
        elif name.lower() == "sparseadam":
            name = "SparseAdam"
        elif name.lower() == "rmsprop":
            name = "RMSprop"
        elif name.lower() == "rprop":
            name = "Rprop"
        elif name.lower() == "adagrad":
            name = "Adagrad"
        elif name.lower == "adadelta":
            name = "Adadelta"
        return name

