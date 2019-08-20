from deeplodocus.utils.namespace import Namespace
from deeplodocus.utils.generic_utils import get_module
from deeplodocus.flags import DEEP_MODULE_TRANSFORMS
from deeplodocus.utils.notification import Notification
from deeplodocus.flags.notif import DEEP_NOTIF_FATAL


class OutputTransformer(Namespace):

    def __init__(self, transform_file=None):
        if transform_file is None:
            transform_file = {
                "name": "EmptyTransformer",
                "transforms": {}
            }
        super(OutputTransformer, self).__init__(transform_file)
        if "transforms" not in self.__dict__:
            raise AttributeError("No 'transforms' entry in given transform file")
        self.load_transform_functions()

    def load_transform_functions(self):
        for transform_name, transform_info in self.transforms.get().items():
            module, module_path = get_module(
                **transform_info.get(ignore="kwargs"),
                browse=DEEP_MODULE_TRANSFORMS
            )
            if module is None:
                Notification(DEEP_NOTIF_FATAL, "Transform %s not found" % transform_name)
            self.transforms.get(transform_name)["method"] = module
            self.transforms.get(transform_name)["module"] = module_path

    def transform(self, data):
        for _, transform in self.transforms.get().items():
            transform.method(data, **transform.kwargs.get())
        return data

