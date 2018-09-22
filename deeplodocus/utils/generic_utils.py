"""
This script contains useful generic functions
"""

import os
import re

def get_file_paths(directory):
    """
    :param dirs: str or list of str: path to directories to get paths from
    :return: list of str: list of paths to every file within the given directories
    """
    paths = []
    for item in os.listdir(directory):
        sub_path = "%s/%s" % (directory, item)
        if os.path.isdir(sub_path):
            paths.extend(get_file_paths(sub_path))
        else:
            paths.extend([sub_path])
    return sorted_nicely(paths)



def sorted_nicely(l):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)



