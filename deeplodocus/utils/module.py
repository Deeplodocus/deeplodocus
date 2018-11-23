import pkgutil

from deeplodocus.utils.notification import Notification
from deeplodocus.utils.flags.notif import *
from deeplodocus.utils.flags.msg import *


def get_module(module, name, silence=False):
    """
    AUTHORS:
    --------

    :author: Alix Leroy
    :author: Samuel Westlake

    DESCRIPTION:
    ------------

    Load a callable in a module

    PARAMETERS:
    -----------

    :param module: The module name
    :param name: The callable name
    :param silence(bool, optional): Whether we want to display the error or not

    RETURN:
    -------

    :return: The loaded module
    """
    local = {"module": None}
    try:
        exec("from %s import %s\nmodule = %s" % (module, name, name), {}, local)
        Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_MODULE_LOADED % (name, module))
    except ImportError as e:
        if not silence:
            Notification(DEEP_NOTIF_ERROR, e)
    return local["module"]


def get_module_browse(path, prefix, name):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Load the desired module
    Works with any python module, deeplodocus module or custom module

    NOTE: Consider the name of the callable to be unique to avoid conflict

    PARAMETERS:
    -----------

    :param path(path): The path to the module
    :param prefix: The prefix to output in front of the module name
    :param name: The name of the callable

    RETURN:
    -------

    :return: The module if loaded, None else
    """
    local = {"module": None}

    # Get the optimizer among the default ones
    for importer, modname, ispkg in pkgutil.walk_packages(path=path,
                                                          prefix=prefix + '.',
                                                          onerror=lambda x: None):
        try:
            exec("from {0} import {1} \nmodule= {2}".format(modname, name, name), {}, local)
            break
        except:
            pass

    return local["module"]