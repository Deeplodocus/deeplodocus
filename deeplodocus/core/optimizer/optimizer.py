from deeplodocus.utils.notification import Notification
from deeplodocus.utils.flags.filter import DEEP_FILTER_OPTIMIZERS
from deeplodocus.utils.flags.notif import DEEP_NOTIF_FATAL
from deeplodocus.utils.generic_utils import get_module
from deeplodocus.utils.namespace import Namespace
from deeplodocus.utils.flags.module import DEEP_MODULE_OPTIMIZERS


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

    def __init__(self, name, module, model_parameters, kwargs=None):
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

        RETURN:
        -------

        :return: None
        """
        self.name = name
        self.module = module
        self.kwargs = Namespace if kwargs is None else kwargs
        self.model_parameters = model_parameters
        self.optimizer = None

    def load(self):
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
        optimizer = get_module(name=self.name,
                               module=self.module,
                               browse=DEEP_MODULE_OPTIMIZERS)
        return optimizer(self.model_parameters, **self.kwargs.get())

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
