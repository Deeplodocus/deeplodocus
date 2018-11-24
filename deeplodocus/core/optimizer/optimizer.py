from deeplodocus.utils.notification import Notification
from deeplodocus.utils.flags.filter import *
from deeplodocus.utils.flags.notif import *
from deeplodocus.utils.dict_utils import check_kwargs
from deeplodocus.utils.generic_utils import get_module
from deeplodocus.utils.namespace import Namespace
from deeplodocus.utils.flags.module import *

class Optimizer(object):
    """
       AUTHORS:
       --------

       :author: Alix Leroy
       :author: Samuel Westlake

       DESCRIPTION:
       ------------

       Optimizer class which loads the optimizer from a PyTorch module or from a custom module
       """

    def __init__(self, model_parameters, config: Namespace):
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Initialize an optimizer by loading it

        PARAMETERS:
        -----------

        :param model_parameters:
        :param config(Namespace):

        RETURN:
        -------

        :return: None
        """
        self.optimizer = self.load(config=config,
                                   model_parameters = model_parameters)
    @staticmethod
    def load(config, model_parameters):
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Load the optimizer in memory

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        optimizer = get_module(config=config,
                               modules=DEEP_MODULE_OPTIMIZERS)
        kwargs = check_kwargs(config.kwargs)
        return optimizer(model_parameters, **kwargs)

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

        :return: The optimizer
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

