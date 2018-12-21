
from deeplodocus.utils.generic_utils import get_module
from deeplodocus.utils.flags.module import DEEP_MODULE_MODELS


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

    def __init__(self, name, module=None, kwargs=None):
        self.name = name
        self.module = module
        self.kwargs = {} if kwargs is None else kwargs
        self.model = None

    def load(self):
        """
        AUTHORS:
        --------

        :author: Samuel Westlake
        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Load and return the model

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        model = get_module(name=self.name, module=self.module, browse=DEEP_MODULE_MODELS)
        return model(**self.kwargs)
