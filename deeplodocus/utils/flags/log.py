from deeplodocus.utils.flags.ext import *
from deeplodocus.utils.flags.path import *

DEEP_LOGS = {"notification": [DEEP_PATH_NOTIFICATION, DEEP_EXT_LOGS],
             "history_train_batches": [DEEP_PATH_HISTORY, DEEP_EXT_CSV],
             "history_train_epochs": [DEEP_PATH_HISTORY, DEEP_EXT_CSV],
             "history_validation": [DEEP_PATH_HISTORY, DEEP_EXT_CSV]}
