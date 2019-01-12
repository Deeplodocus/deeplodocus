from deeplodocus.utils.flags.filter import DEEP_FILTER_OPTIMIZERS
from deeplodocus.utils.generic_utils import get_module
from deeplodocus.utils.flags.module import DEEP_MODULE_OPTIMIZERS
from deeplodocus.utils.notification import Notification
from deeplodocus.utils.flags.notif import *


def load_optimizer(name, module, model_parameters, kwargs):
    optimizer = get_module(name=name,
                           module=module,
                           browse=DEEP_MODULE_OPTIMIZERS)

    class Optimizer(optimizer):

        def __init__(self, name, module, model_parameters, kwargs_dict, **kwargs):
            super(Optimizer, self).__init__(model_parameters, **kwargs)
            self.name = name
            self.module = module
            self.kwargs = kwargs_dict

        def summary(self):
            Notification(DEEP_NOTIF_INFO, '================================================================')
            Notification(DEEP_NOTIF_INFO, "OPTIMIZER : %s from %s" % (self.name, self.module))
            Notification(DEEP_NOTIF_INFO, '================================================================')
            for key, value in self.kwargs.items():
                if key != "name":
                    Notification(DEEP_NOTIF_INFO, "%s : %s" % (key, value))
            Notification(DEEP_NOTIF_INFO, "")

    return Optimizer(name,
                     module,
                     model_parameters,
                     kwargs.get(), **kwargs.get())


# Currently not in use
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
