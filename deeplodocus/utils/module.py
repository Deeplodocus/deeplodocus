import pkgutil

def get_module(path, prefix, name):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Load the desired module
    Works with any python module, deeplodocus module or custom module

    PARAMETERS:
    -----------

    :param path(path): The path to the module
    :param prefix: The prefix to output in front of the module name
    :param name: The name of the module

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