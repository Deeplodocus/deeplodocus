"""
This script contains useful generic functions
"""

import os
import re





def sorted_nicely(l):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)



def is_string_an_integer(string:str)->bool:
    try :
        int(string)
    except:
        return False

    return True

