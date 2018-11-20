from deeplodocus.utils.flags.ext import DEEP_EXT_YAML
from deeplodocus.utils.flags import *

#
# DEEP CONFIG FLAGS
#

DEEP_CONFIG_DIVIDER = "/"

DEEP_CONFIG_PROJECT = "project"
DEEP_CONFIG_MODEL = "model"
DEEP_CONFIG_TRAINING = "training"
DEEP_CONFIG_DATA = "data"
DEEP_CONFIG_TRANSFORM = "transform"
DEEP_CONFIG_OPTIMIZER = "optimizer"
DEEP_CONFIG_METRICS = "metrics"
DEEP_CONFIG_LOSS = "losses"
DEEP_CONFIG_HISTORY = "history"

DEEP_CONFIG_SECTIONS = [DEEP_CONFIG_PROJECT,
                        DEEP_CONFIG_MODEL,
                        DEEP_CONFIG_TRAINING,
                        DEEP_CONFIG_DATA,
                        DEEP_CONFIG_TRANSFORM,
                        DEEP_CONFIG_OPTIMIZER,
                        DEEP_CONFIG_METRICS,
                        DEEP_CONFIG_LOSS,
                        DEEP_CONFIG_HISTORY]


DEEP_CONFIG = {"project": {"name": "project",
                           "cv_library": 1,
                           "logs": {"history_train_batches": True,
                                    "history_train_epochs": True,
                                    "history_validation": True,
                                    "notification": True},
                           "on_wake": None},
               "model": {"name": None,
                         "module": None},
               "history" : {"verbose" : 2,
                            "memorize" : 1},
               "training" : {"num_epochs" : 10,
                             "initial_epoch" : 0,
                             "shuffle" : 2,
                             "save_condition" : 1},
               "data": {"dataloader": {"batch_size": 32,
                                       "num_workers": 1},
                        "dataset": {"train": {"inputs": None,
                                              "labels": None,
                                              "additional_data": None},
                                    "validation": {"inputs": None,
                                                   "labels": None,
                                                   "additional_data": None},
                                    "test": {"inputs": None,
                                             "labels": None,
                                             "additional_data": None}}}}

DEEP_CONFIG_FILES = {item: "%s%s" % (item, DEEP_EXT_YAML) for item in DEEP_CONFIG_SECTIONS}
