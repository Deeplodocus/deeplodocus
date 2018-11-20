from deeplodocus.utils.flags.ext import DEEP_EXT_YAML
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


DEEP_CONFIG = {"project": ["name",
                           "cv_library",
                           {"logs": ["keep"]},
                           "on_wake"],
               "model": ["module", "name"],
               "training" : ["num_epochs",
                             "initial_epoch",
                             "shuffle",
                             "verbose",
                             "save_condition",
                             "memorize"],
               "data": [{"dataloader": ["batch_size",
                                        "num_workers"]},
                        {"dataset": [{"train": ["inputs",
                                                "labels",
                                                "additional_data"],
                                      "validation": ["inputs",
                                                     "labels",
                                                     "additional_data"],
                                      "test": ["inputs",
                                               "labels",
                                               "additional_data"]}]}]}

DEEP_CONFIG_FILES = {item: "%s%s" % (item, DEEP_EXT_YAML) for item in DEEP_CONFIG_SECTIONS}
