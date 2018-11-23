import os
import __main__

DEEP_PATH_NOTIFICATION = r"%s/logs" % os.path.dirname(os.path.abspath(__main__.__file__))
DEEP_PATH_RESULTS = r"%s/results" % os.path.dirname(os.path.abspath(__main__.__file__))
DEEP_PATH_HISTORY = r"%s/results/history" % os.path.dirname(os.path.abspath(__main__.__file__))
DEEP_PATH_SAVE_MODEL = r"%s/results/models" % os.path.dirname(os.path.abspath(__main__.__file__))

DEEP_PATH_MODULES = "deeplodocus.core.project.deep_structure.modules"
DEEP_PATH_MODELS = "%s.models" % DEEP_PATH_MODULES
DEEP_PATH_OPTIMIZERS = "%s.optimizers" % DEEP_PATH_MODULES
