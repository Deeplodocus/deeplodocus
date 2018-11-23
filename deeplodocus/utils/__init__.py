import __main__

import os


def get_main_path():
    """
    :return:
    """
    return os.path.dirname(os.path.abspath(__main__.__file__))
