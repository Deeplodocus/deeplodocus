"""
This script contains useful generic functions
"""
import re
import pkgutil
import __main__
import random
import string

from deeplodocus.utils.flags.ext import *
from deeplodocus.utils.flags.notif import *
from deeplodocus.utils.flags.dtype import *
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
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Check whether a string is an integer or not

    PARAMETERS:
    -----------

    :param string(str): The string to analyze

    RETURN:
    -------

    :return (bool): Whether the string is an integer or not
    """
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


def get_specific_module(name, module, silence=False):
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
    except ImportError as e:
        if not silence:
            Notification(DEEP_NOTIF_ERROR, str(e))
    return local["module"]


def get_module(name, module=None, browse=None):
    """
    AUTHORS:
    --------

    :author: Alix Leroy
    :author: Samuel Westlake

    DESCRIPTION:
    ------------

    Get a module from either a direct import or a folder browsing

    PARAMETERS:
    -----------

    :param name: str: the name of the object to load
    :param module: str: the name of the specific module
    :param browse: dict: a DEEP_MODULE dictionary to browse through

    RETURN:
    -------

    :return module(callable): The loaded module
    """
    if module is not None:
        return get_specific_module(name, module)
    elif browse is not None:
        return browse_module(name, browse)
    else:
        return None


def browse_module(name, modules):
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

    :param modules: dict
    :param name: The name of the callable

    RETURN:
    -------

    :return: The module if loaded, None else
    """
    list_modules = []
    # For all the given modules
    for key, value in modules.items():
        # For all the sub-modules available in the main module
        for importer, modname, ispkg in pkgutil.walk_packages(path=value["path"],
                                                              prefix=value["prefix"] + '.',
                                                              onerror=lambda x: None):

            # Fix the loading a of useless torch module(temporary)
            if modname == "torch.nn.parallel.distributed_c10d":
                continue

            # Try to get the module
            module = get_specific_module(name, modname, silence=True)
            # If the module exists add it to the list
            if module is not None:
                list_modules.append(module)

    list_modules = remove_duplicates(items=list_modules)
    if len(list_modules) == 0:
        Notification(DEEP_NOTIF_FATAL, "Couldn't find the module '%s' anywhere.")
    elif len(list_modules) == 1:
        return list_modules[0]
    else:
        return select_module(list_modules, name)


def remove_duplicates(items: list):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Remove the duplicate items in a list

    PARAMETERS:
    -----------

    :param items(list): The list of items

    RETURN:
    -------

    :return (list): The lis of items without the duplicates
    """
    return list(set(items))


def select_module(list_modules: list, name: str):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Select the desired module among a list of similar names

    PARAMETERS:
    -----------

    :param list_modules(list): List containing the similar modules
    :param name(str): Name of the module

    RETURN:
    -------

    :return: The desired module
    """
    Notification(DEEP_NOTIF_WARNING, "The module '%s' was found in multiple locations :" % name)
    # Print the list of modules and their corresponding indices
    for i, module in enumerate(list_modules):
        Notification(DEEP_NOTIF_WARNING, "%i : %s" % (i, module))

    # Prompt the user to pick on from the list
    response = -1
    while (response < 0 or response >= len(list_modules)):
        response = Notification(DEEP_NOTIF_INPUT, "Which one would you prefer to use ? (Pick a number)").get()

        # Check if the response is an integer
        if is_string_an_integer(response) is False:
            response = -1
        else:
            response = int(response)

    return list_modules[response]


def generate_random_alphanumeric(size: int = 16):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Generate a string of alphanumeric characters of a specific size
    The default size is 16 characters

    PARAMETERS:
    -----------

    :param size(int): The size of the alphanumeric string

    RETURN:
    -------

    :return (string): The random alphanumeric string
    """

    return''.join(random.choices(string.ascii_letters + string.digits, k=size))
