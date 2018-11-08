import torch


from deeplodocus.utils.notification import Notification
from deeplodocus.utils.flags import *
class Optimizer(object):

    def __init__(self, name:str,
                 params,
                 lr:float=0.001,
                 momentum:int=0.0,
                 max_iter:int = 20,
                 max_eval:int=None,
                 betas=(0.9, 0.999),

                 weight_decay=0,
                 write_logs: bool = True):

        self.write_logs=write_logs

        if isinstance(name, str):
            self.optimizer = self.__check_optimizer(name, params,  **kwargs)
        else:
            Notification(DEEP_NOTIF_FATAL, "The following name is not a string : " + str(name), write_logs=self.write_logs)



    def __check_optimizer(self, name:str, params, **kwargs):
        optimizer = None

        if name.lower() == "sgd":
            optimizer = torch.optim.SGD(params, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False)
        elif name.lower == "adam":
            optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        elif name.lower() == "adamax":
            optimizer = torch.optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

        # Averaged Stochastic Gradient Descent
        elif name.lower == "asgd":
            optimizer = torch.optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
        elif name.lower() == "lbfgs":
            optimizer = torch.optim.LBFGS(params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)
        elif name.lower() =="sparseadam":
            optimizer = torch.optim.SparseAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)

        elif name.lower() == "rmsprop":
            optimizer = torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        elif name.lower() == "rprop":
            optimizer = torch.optim.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
        elif name.lower() == "adagrad":
            optimizer = torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
        elif name.lower == "adadelta":
            optimizer = torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
        else:
            Notification(DEEP_NOTIF_FATAL, "The following optimizer does no exist, please check the documentation : " + str(name), write_logs=self.write_logs)

        return optimizer

