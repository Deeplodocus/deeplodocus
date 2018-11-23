
from deeplodocus.utils.dict_utils import check_kwargs
from deeplodocus.utils.generic_utils import get_module


class Model(object):

    def __init__(self, config):
        self.model = None
        self.config = config
        self.load_model()

    def load_model(self):
        """
        :return:
        """
        model = get_module(module=self.config.module,
                           name=self.config.name)
        kwargs = check_kwargs(self.config.kwargs)
        self.model = model(**kwargs)

    def get(self):
        """
        :return:
        """
        return self.model
