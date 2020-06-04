from deeplodocus.flags import DEEP_FILTER_OPTIMIZERS
from deeplodocus.utils.generic_utils import get_module
from deeplodocus.flags import DEEP_MODULE_OPTIMIZERS
from deeplodocus.utils.notification import Notification

# Deeplodocus flags
from deeplodocus.flags import *


def load_optimizer(name, module, model, kwargs, param_groups=None, verbose=True):

    pgs = param_groups
    if param_groups is not None:
        param_groups = make_param_groups(param_groups, model, verbose=verbose)

    optimizer, module = get_module(
        name=name,
        module=module,
        browse=DEEP_MODULE_OPTIMIZERS
    )

    class Optimizer(optimizer):

        def __init__(self, name, module, parameters, **kwargs):
            super(Optimizer, self).__init__(parameters, **kwargs)
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
        optimizer = Optimizer(
            name=name,
            module=module,
            parameters=param_groups[0]["params"],
            **{**kwargs.get(), **param_groups[0]["kwargs"]}
        )
        for group in param_groups[1:]:
            optimizer.add_param_group({"params": group["params"], **{**kwargs.get(), **group["kwargs"]}})

        # Print summary of the parameter groups
        if verbose:
            kwargs = [{k: v for k, v in sorted(group.items()) if k != "params"} for group in optimizer.param_groups]
            for i, (param_group, pg) in enumerate(zip(param_groups, pgs)):
                Notification(
                    DEEP_NOTIF_INFO,
                    "Parameter Group %s : %s : %s sets : kwargs=%s" % (
                        str(i).ljust(3),
                        ("%s" % pg.condition).ljust(max([len(str(i.condition)) for i in pgs])),
                        str(len(param_group["params"])).rjust(4),
                        kwargs[i]
                    )
                )
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
    return groups
