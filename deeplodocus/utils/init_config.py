from os import makedirs

from deeplodocus.utils.flags.config import DEEP_CONFIG
from deeplodocus.utils.flags.ext import DEEP_EXT_YAML
from deeplodocus.utils.namespace import Namespace

config = Namespace(DEEP_CONFIG)
config_dir = "/home/samuel/config_new"
makedirs(config_dir, exist_ok=True)
for key, namespace in config.get().items():
    if isinstance(namespace, Namespace):
        namespace.save("%s/%s%s" % (config_dir, key, DEEP_EXT_YAML))
