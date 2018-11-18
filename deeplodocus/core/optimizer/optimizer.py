import torch


from deeplodocus.utils.notification import Notification
from deeplodocus.utils.flags import *


class Optimizer(object):

    def __init__(self, name:str,
                 params,
                 write_logs: bool = True,
                 **kwargs):

        self.write_logs=write_logs

        if isinstance(name, str):
            name = self.__format_name(name)
            self.optimizer = self.__select_optimizer(name, params,  **kwargs)
        else:
            Notification(DEEP_NOTIF_FATAL, "The following name is not a string : " + str(name), write_logs=self.write_logs)

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
            Notification(DEEP_NOTIF_FATAL, "The following optimizer is not allowed : " + str(name), write_logs=self.write_logs)

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
        local = {"optimizer" : None}
        exec("import torch \noptimizer = torch.optim.{0}".format(name), {}, local)
        print(kwargs)
        optimizer = local["optimizer"](params, **kwargs)

        return optimizer

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
