import __main__
import os


def get_main_path():
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Get the path to the main running file.

    PARAMETERS:
    -----------

    None

    RETURN:
    -------

    :return: The path to the main file
    """
    return os.path.dirname(os.path.abspath(__main__.__file__))