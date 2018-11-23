import os
import __main__

from deeplodocus.utils.main_utils import get_main_path

DEEP_PATH_NOTIFICATION = r"%s/logs" % get_main_path()
DEEP_PATH_HISTORY = r"%s/results/history" % get_main_path()
DEEP_PATH_SAVE_MODEL = r"%s/results/models" % get_main_path()


DEEP_PATH_MODULES = ".modules"                                      # Relative path to the "modules" module
DEEP_PATH_MODELS = "%s.models" % DEEP_PATH_MODULES
DEEP_PATH_OPTIMIZERS = "%s.optimizers" % DEEP_PATH_MODULES
