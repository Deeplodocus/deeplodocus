from deeplodocus.flags import DEEP_FILTER_OPTIMIZERS
from deeplodocus.utils.generic_utils import get_module
from deeplodocus.flags import DEEP_MODULE_OPTIMIZERS
from deeplodocus.utils.notification import Notification

# Deeplodocus flags
from deeplodocus.flags import *

def load_optimizer(name, module, model_parameters, kwargs):
    optimizer, module = get_module(
        name=name,
        module=module,
        browse=DEEP_MODULE_OPTIMIZERS
    )

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

    return Optimizer(
        name,
        module,
        model_parameters,
        kwargs.get(), **kwargs.get()
    )
