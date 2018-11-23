import operator
from collections import defaultdict

from deeplodocus.utils.generic_utils import get_int_or_float
from deeplodocus.utils.flags import *


def check_kwargs(namespace):
    """
    Author: Samuel Westlake
    :param namespace: Namespace object
    :return: prepare namespace.kwargs for use
    """
    if namespace.kwargs is None:
        return {}
    else:
        return convert_string_to_number(namespace.kwargs.get())


def append(item1, item2):
    """
    Author: SW
    :param item1: base item
    :param item2: item to be appended to the base
    :return: item1 with second item appended
    """
    if not isinstance(item1, list):
        item1 = [item1]
    item1.append(item2)
    return item1


def convert_string_to_number(dictionary: dict):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Convert strings in a dictionary to float or integers

    PARAMETERS:
    -----------

    :param dictionary -> dictionary: The dictionary to convert

    RETURN:
    -------

    :return dict->dict: The converted dictionary
    """
    for key, value in dictionary.items():
        if isinstance(value, bool):
            continue
        dtype = get_int_or_float(value)
        # Float
        if dtype == DEEP_TYPE_FLOAT:
            dictionary[key] = float(value)
        # Integer
        elif dtype == DEEP_TYPE_INTEGER:
            dictionary[key] = int(value)
    return dictionary


def merge_sum_dict(*dicts):
    """
    AUTHORS:
    --------

    :author: https://stackoverflow.com/questions/10461531/merge-and-sum-of-two-dictionaries
    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Merge and sum multiple dictionaries

    PARAMETERS:
    -----------

    :param dicts->List[dict]: A list of dictionaries

    RETURN:
    -------

    :return: The merged and summed dictionary
    """
    #
    ret = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            ret[k] += v
    return dict(ret)


def sum_dict(dictionary: dict):
    """
    Author: Alix Leroy, SW
    Returns the sum of all the values in a dictionary
    :param dictionary: dict: input dictionary of float/int
    :return: float: sum of the values in the dictionary
    """
    return sum(list(dictionary.values()))


def like(dictionary: dict, value=None):
    """
    :param dictionary:
    :param value: value to initialise each key with
    :return:
    """
    new_dictionary = {}
    for key in dictionary:
        try:
            new_dictionary[key] = value.copy()
        except AttributeError:
            new_dictionary[key] = value
    return new_dictionary


def mean(dictionary: dict):
    """
    Author: SW
    Returns the mean of each value in the dictionary for each key
    :param dictionary: dict: an input dictionary of iterables
    :return: dict: dictionary with the mean of all values
    """
    for key, value in dictionary.items():
        dictionary[key] = sum(value) / len(value)
    return dictionary


def apply(dictionary1: dict, dictionary2: dict, op: str, error_ok: bool=False):
    """
    Author: SW
    Apply to the values of dictionary1 the corresponding values from dictionary2 by the given operator
    :param dictionary1: dict: the base dictionary
    :param dictionary2: dict: the dictionary to apply
    :param op: str: the chosen operation
    :param error_ok: bool: indicates if any KeyErrors should be ignored (i.e. if dict2 does not contain all keys)
    :return: dictionary1: dict: the base dictionary after applying the second dictionary by the given operator
    """
    for key, value, in dictionary1.items():
        if error_ok:
            try:
                dictionary1[key] = OPERATIONS[op](value, dictionary2[key])
            except KeyError:
                pass
        else:
            dictionary1[key] = OPERATIONS[op](value, dictionary2[key])
    return dictionary1


def apply_weight(loss_dictionary: dict, losses: dict):
    """
    Author: SW
    Apply the weight for each loss to a dictionary of computed loss values
    :param loss_dictionary: dict: the name and value of each loss function
    :param losses: dict: dict: the name and loss objects
    :return: loss_dictionary: dict: loss_dictionary with weights applied
    """
    for key, value in loss_dictionary.items():
        loss_dictionary[key] = value * losses[key].get_weight()
    return loss_dictionary


OPERATIONS = {"*": operator.mul,
              "-": operator.sub,
              "+": operator.add,
              "/": operator.truediv,
              "^": operator.pow,
              "append": append}

if __name__ == "__main__":
    # For testing
    d = {"a": [1,2,3], "b": [5, 43, 1, 65, 312]}
    print(mean(d))
    d1 = {"a": 5, "b": 6, "c": 100, "d": -7}
    d2 = {"a": 7, "b": 2, "c": 20, "d": 2}
    print(apply(d1, d2, "append"))
    d3 = like(d2, [])
    d3["a"].append(4)
    print(d3)