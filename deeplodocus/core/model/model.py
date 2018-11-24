
from deeplodocus.utils.dict_utils import check_kwargs
from deeplodocus.utils.generic_utils import get_module
from deeplodocus.utils.namespace import Namespace
from deeplodocus.utils.flags.module import *

class Model(object):
    """
    AUTHORS:
    --------

    :author: Samuel Westlake
    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Model class containing the model
    """

    def __init__(self, config: Namespace):
        self.model = self.load(config)

    @staticmethod
    def load(config: Namespace):
        """
        AUTHORS:
        --------

        :author: Samuel Westlake
        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Load the model

        PARAMETERS:
        -----------

        :param config(Namespace): The parameters from the model config file

        RETURN:
        -------

        :return: None
        """
        model = get_module(config=config,
                           modules=DEEP_MODULE_MODELS)
        kwargs = check_kwargs(config.kwargs)
        return model(**kwargs)

    def get(self):
        """
        AUTHORS:
        --------

        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Get the model

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        return self.model
