import torch
import pkgutil


from deeplodocus.utils.notification import Notification
from deeplodocus.utils.flags import *
from deeplodocus.utils.main_utils import *
from deeplodocus.utils.module import get_module


class Optimizer(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Optimizer class which loads the optimizer from a PyTorch module or from a custom module
    """

    def __init__(self, name: str, params, **kwargs):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize an Optimizer instance

        PARAMETERS:
        -----------

        :param name->str: The name of the optimizer
        :param params: Parameters to optimize
        :param kwargs: Arguments of the optimizer

        RETURN:
        -------

        :return: None
        """

        if isinstance(name, str):
            name = self.__format_name(name)
            self.optimizer = self.__select_optimizer(name, params,  **kwargs)
        else:
            Notification(DEEP_NOTIF_FATAL, "The following name is not a string : " + str(name))

    @staticmethod
    def __format_name(name: str):
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

        # Filter illegal optimizers
        if name.lower() in DEEP_FILTER_OPTIMIZERS:
            Notification(DEEP_NOTIF_FATAL, "The following optimizer is not allowed : %s" % name)
        # Format already known
        if name.lower() == "sgd":
            name = "SGD"
        elif name.lower == "adam":
            name = "Adam"
        elif name.lower() == "adamax":
            name = "Adamax"
        # Averaged Stochastic Gradient Descent
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

    @staticmethod
    def __select_optimizer(name: str, params, **kwargs):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Select the desired optimizer

        PARAMETERS:
        -----------

        :param name->str: The name of the optimizer
        :param params: The parameters of the model
        :param kwargs: The arguments of the optimizer

        RETURN:
        -------

        :return optimizer: The optimizer
        """
        optimizer = get_module(module=torch.optim.__name__,
                               name=name)
        if optimizer is None:
            optimizer = get_module(module=DEEP_PATH_OPTIMIZERS,
                                   name=name)
        if optimizer is None:
            Notification(DEEP_NOTIF_FATAL, "The following optimizer could not be loaded neither from the standard nor from the custom ones : " + str(name))
        return optimizer(params, **kwargs)

    def get(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the optimizer

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return self.optimizer: The optimizer
        """
        return self.optimizer
