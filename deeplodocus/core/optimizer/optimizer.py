import torch
import pkgutil


from deeplodocus.utils.notification import Notification
from deeplodocus.utils.flags import *
from deeplodocus.utils.main_utils import *


class Optimizer(object):

    def __init__(self, name:str,
                 params,
                 **kwargs):


        if isinstance(name, str):
            name = self.__format_name(name)
            self.optimizer = self.__select_optimizer(name, params,  **kwargs)
        else:
            Notification(DEEP_NOTIF_FATAL, "The following name is not a string : " + str(name))

    def __format_name(self, name:str):
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
            Notification(DEEP_NOTIF_FATAL, "The following optimizer is not allowed : " + str(name))

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
        elif name.lower() =="sparseadam":
            name = "SparseAdam"
        elif name.lower() == "rmsprop":
            name = "RMSprop"
        elif name.lower() == "rprop":
            name = "Rprop"
        elif name.lower() == "adagrad":
            name = "Adagrad"
        elif name.lower == "adadelta":
            name ="Adadelta"
        else:
            # Keep the given entry
            pass
        return name


    def __select_optimizer(self, name:str, params, **kwargs):
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
        #Method 1
        #local = {"optimizer" : None}
        #exec("import torch \noptimizer = torch.optim.{0}".format(name), {}, local)
        #optimizer = local["optimizer"](params, **kwargs)

        #Method2
        #optimizer = getattr(torch.optim, name)

        # Method3 (accepts custom optimizers)
        local = {"optimizer": None}

        # Get the transform method among the default ones
        for importer, modname, ispkg in pkgutil.walk_packages(path=torch.optim.__path__,
                                                              prefix=torch.optim.__name__ + '.',
                                                              onerror=lambda x: None):
            try:
                exec("from {0} import {1} \noptimizer= {2}".format(modname, name, name), {}, local)
                break
            except:
                pass

        # Get the optimizer among the custom ones
        if local["optimizer"] is None:
            for importer, modname, ispkg in pkgutil.walk_packages(path=[get_main_path() + "/modules/optimizers"],
                                                                  prefix = "modules.optimizers.",
                                                                  onerror=lambda x: None):
                try:
                    exec("from {0} import {1} \noptimizer= {2}".format(modname, name, name), {}, local)
                    break
                except:
                    pass

        # If neither a standard not a custom transform is loaded
        if local["optimizer"] is None:
            Notification(DEEP_NOTIF_FATAL, "The following optimizer could not be loaded neither from the standard nor from the custom ones : " + str(name))

        return local["optimizer"](params, **kwargs)

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
