from deeplodocus.flags import DEEP_FILTER_OPTIMIZERS
from deeplodocus.utils.generic_utils import get_module
from deeplodocus.flags import DEEP_MODULE_OPTIMIZERS
from deeplodocus.utils.notification import Notification

# Deeplodocus flags
from deeplodocus.flags import *


def load_optimizer(name, module, model, kwargs, param_groups=None, verbose=True):

    if param_groups is not None:
        param_groups = make_param_groups(param_groups, model, verbose=verbose)

    optimizer, module = get_module(
        name=name,
        module=module,
        browse=DEEP_MODULE_OPTIMIZERS
    )

    class Optimizer(optimizer):

        def __init__(self, name, module, model_parameters, **kwargs):
            super(Optimizer, self).__init__(model_parameters, **kwargs)
            self.name = name
            self.module = module

        def summary(self):
            Notification(DEEP_NOTIF_INFO, '================================================================')
            Notification(DEEP_NOTIF_INFO, "OPTIMIZER : %s from %s" % (self.name, self.module))
            Notification(DEEP_NOTIF_INFO, '================================================================')
            Notification(DEEP_NOTIF_INFO, "SORRY, NOT IMPLEMENTED YET")

    if param_groups is None:
        return Optimizer(name, module, model.parameters(), **kwargs.get())
    else:
        optimizer = Optimizer(name, module, param_groups[0]["params"], **param_groups[0]["kwargs"])
        for group in param_groups[1:]:
            optimizer.add_param_group({"params": group["params"], **group["kwargs"]})
        return optimizer


def make_param_groups(param_groups, model, verbose=True):
    if verbose:
        Notification(DEEP_NOTIF_INFO, "Assigning parameter groups")
    # Convert pg.condition from string to lambda
    conditions = []
    for i, pg in enumerate(param_groups):
        local = {"condition": None}
        exec("condition = %s" % pg.condition, {}, local)
        conditions.append(local["condition"])
    # Assign each set of parameters into corresponding group (by name, given conditions)
    groups = [{"params": [], "kwargs": vars(param_groups[i].kwargs)} for i in range(len(param_groups))]
    for key, params in dict(model.named_parameters()).items():
        for i, condition in enumerate(conditions):
            if condition is None or condition(key):
                groups[i]["params"].append(params)
                if verbose:
                    Notification(DEEP_NOTIF_INFO, "  %s\t-> group %i" % (key, i))
                break
    # Print summary of the parameter groups
    if verbose:
        for i, (group, pg) in enumerate(zip(groups, param_groups)):
            Notification(
                DEEP_NOTIF_INFO,
                "Parameter Group %s : %s : %s sets : kwargs=%s" % (
                    str(i).ljust(3),
                    ("%s" % pg.condition).ljust(max([len(str(pg.condition)) for pg in param_groups])),
                    str(len(group["params"])).rjust(4),
                    group["kwargs"]
                )
            )
    return groups
