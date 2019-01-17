"""
This script contains useful generic functions
"""
import re
import pkgutil
import random
import string
from typing import List
from typing import Union
from typing import Optional

from deeplodocus.utils.flags.ext import *
from deeplodocus.utils.flags.notif import *
from deeplodocus.utils.flags.dtype import *
from deeplodocus.utils.notification import Notification
from deeplodocus.utils.flag import Flag
from deeplodocus.utils.namespace import Namespace


def convert(value, d_type=None):
    """
    Convert a value or list of values to data type in order of preference: (float, bool, str)
    :param value: value to convert
    :param d_type: data type to convert to
    :return: converted value
    """
    if value is None:
        return None
    elif d_type is None:
        if isinstance(value, list):
            return [convert(item) for item in value]
        else:
            new_value = convert2float(value)
            if new_value is not None:
                if round(new_value, 0) == new_value:
                    return int(new_value)
                else:
                    return new_value
            new_value = convert2bool(value)
            if new_value is not None:
                return new_value
            new_value = convert2bool(value)
            if new_value is not None:
                return new_value
            else:
                return str(value)
    elif d_type is str:
        return str(value)
    elif d_type is int:
        return convert2int(value)
    elif d_type is float:
        return convert2float(value)
    elif d_type is bool:
        return convert2bool(value)
    elif d_type is dict:
        try:
            return convert_namespace(value)
        except AttributeError:
            return None
    elif isinstance(d_type, dict):
        new_value = {}
        for key, item in d_type.items():
            try:
                new_value[key] = convert(value[key], d_type=item)
            except KeyError:
                new_value[key] = None
            except TypeError:
                return None
        return Namespace(new_value)
    elif isinstance(d_type, list):
        value = value if isinstance(value, list) else [value]
        new_value = []
        for item in value:
            new_item = convert(item, d_type[0])
            if new_item is None:
                return None
            else:
                new_value.append(new_item)
        return new_value


def convert2int(value):
    try:
        return int(eval(value))
    except (ValueError, TypeError, SyntaxError, NameError):
        pass
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def convert2float(value):
    try:
        return float(eval(value))
    except (ValueError, TypeError, SyntaxError, NameError):
        pass
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def convert2bool(value):
    try:
        return bool(value)
    except TypeError:
        return None


def convert_namespace(namespace):
        """
        AUTHORS:
        --------
        :author: Samuel Westlake

        DESCRIPTION:
        ------------
        Converts each value in a namespace to the most appropriate data type

        PARAMETERS:
        -----------
        :param namespace: a given namespace to convert the values of

        RETURN:
        -------
        :return: the namespace with each value converted to a sensible data type
        """
        for key, value in namespace.get().items():
            namespace.get()[key] = convert(value)
        return namespace


def sorted_nicely(l):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """
    to_convert = lambda text: int(text) if text.isdigit() else text
    alpha_num_key = lambda key: [to_convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alpha_num_key)


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


def get_module(name: str, module=None, browse=None) -> Union[callable, None]:
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

    :return module(Union[callable, None]): The loaded module
    """
    if module is not None:
        return get_specific_module(name, module)
    elif browse is not None:
        return browse_module(name, browse)
    else:
        return None


def browse_module(name, modules) -> callable:
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

    :return (callable): The loaded module
    """
    list_modules = []
    # For all the given modules
    for key, value in modules.items():
        # For all the sub-modules available in the main module
        for importer, modname, ispkg in pkgutil.walk_packages(path=value["path"],
                                                              prefix=value["prefix"] + '.',
                                                              onerror=lambda x: None):

            # TODO : Remove when torch module is updated to 1.0.1+
            # Fix the loading a of useless torch module(temporary)
            if modname == "torch.nn.parallel.distributed_c10d":
                continue

            # Try to get the module
            module = get_specific_module(name, modname, silence=True)
            # If the module exists add it to the list
            if module is not None:
                list_modules.append(module)

    # Remove modules found multiple times
    list_modules = remove_duplicates(items=list_modules)

    # If not module was found
    if len(list_modules) == 0:
        Notification(DEEP_NOTIF_FATAL, "Couldn't find the module '%s' anywhere." %name)

    # If only one module was found
    elif len(list_modules) == 1:
        return list_modules[0]

    # If more than one module was found we ask the user to select the good one
    else:
        return select_module(list_modules, name)


def remove_duplicates(items: list) -> list:
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

    :return (list): The list of items without the duplicates
    """
    return list(set(items))


def select_module(list_modules: list, name: str) -> callable:
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

    :return: The desired module(callable)
    """
    Notification(DEEP_NOTIF_WARNING, "The module '%s' was found in multiple locations :" % name)
    # Print the list of modules and their corresponding indices
    for i, module in enumerate(list_modules):
        Notification(DEEP_NOTIF_WARNING, "%i : %s" % (i, module))

    # Prompt the user to pick on from the list
    response = -1
    while response < 0 or response >= len(list_modules):
        response = Notification(DEEP_NOTIF_INPUT, "Which one would you prefer to use ? (Pick a number)").get()

        # Check if the response is an integer
        if is_string_an_integer(response) is False:
            response = -1
        else:
            response = int(response)

    return list_modules[response]


def generate_random_alphanumeric(size: int = 16) -> str:
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


def get_corresponding_flag(flag_list: List[Flag], info : Union[str, int, Flag], fatal: bool =True, default: Optional[Flag]= None) -> Union[Flag, None]:
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Browse the wanted flag among a list
    If no flag corresponds, raise a DeepError

    PARAMETERS:
    -----------

    :param flag_list (List[Flag]) : The list of flag to browse in
    :param name (Union[str, int, Flag]): Info (name, index or full Flag) of the flag to search
    :param fatal(bool, Optional): Whether to raise a DeepError if no flag is found or not

    RETURN:
    -------

    :return : The corresponding flag
    """

    # Search in the flag list
    for flag in flag_list:
        if flag.corresponds(info=info) is True:
            return flag

    # If no flag is found
    if default is not None:
        Notification(DEEP_NOTIF_WARNING,
                     "The following flag does not exist : %s, the default one %s has been selected instead" % (str(info), default.get_description()))

        return default

    # If no default
    if fatal is True:
        Notification(DEEP_NOTIF_FATAL, "No flag with the info '%s' was found in the following list : %s" %(str(info), str([flag.description for flag in flag_list])))
    else:
        return None
