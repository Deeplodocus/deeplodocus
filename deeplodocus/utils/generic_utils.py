"""
This script contains useful generic functions
"""
import re
import os
import pkgutil
import __main__

from deeplodocus.utils.flags.ext import *
from deeplodocus.utils.flags.notif import *
from deeplodocus.utils.flags.msg import *
from deeplodocus.utils.flags.type import *
from deeplodocus.utils.notification import Notification


def sorted_nicely(l):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def is_string_an_integer(string: str) -> bool:
    try:
        int(string)
        return True
    except ValueError:
        return False


def get_int_or_float(data):
    """
    AUTHORS:
    --------

    author: Alix Leroy

    DESCRIPTION:
    ------------

    Check whether the data is an integer or a float

    PARAMETERS:
    -----------

    :param data: The data to check

    RETURN:
    -------

    :return:  The integer flag of the corresponding type or False if the data isn't a number
    """
    if isinstance(data, list):
        return False
    elif isinstance(data, tuple):
        return False
    try:
        number_as_float = float(data)
        number_as_int = int(number_as_float)
        return DEEP_TYPE_INTEGER if number_as_float == number_as_int else DEEP_TYPE_FLOAT
    except ValueError:
        return False


def is_np_array(data):
    """
    AUTHORS:
    --------

    author: Alix Leroy

    DESCRIPTION:
    ------------

    Check whether the data is an numpy array or not

    PARAMETERS:
    -----------

    :param data: The data to check

    RETURN:
    -------

    :return:  Whether the data is a numpy array or not
    """
    try:
        if data.endswith(DEEP_EXT_NPY) or data.endswith(DEEP_EXT_NPZ):
            return True
    except:
        return False


def get_module(module, name, silence=False):
    """
    Author: Samuel Westlake
    :param module: str: path to the module (separated by '.')
    :param name: str: name of the item to be imported
    :param silence: bool: whether or not to silence any ImportError
    :return:
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
