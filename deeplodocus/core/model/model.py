from deeplodocus.utils.flags.module import DEEP_MODULE_MODELS
from deeplodocus.utils.generic_utils import get_module
from deeplodocus.utils.namespace import Namespace


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

    def __init__(self, name, input_size=None, module=None, kwargs=None):
        self.name = name
        self.module = module
        self.input_size = input_size
        self.kwargs = Namespace() if kwargs is None else kwargs
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
        return model(**self.kwargs.get())
