from deeplodocus.utils.flags.ext import DEEP_EXT_CSV, DEEP_EXT_LOGS
from deeplodocus.utils import get_main_path

DEEP_LOG_NOTIFICATION = "notification"
DEEP_LOG_HISTORY_TRAIN_BATCHES = "history_train_batches"
DEEP_LOG_HISTORY_TRAIN_EPOCHS = "history_train_epochs"
DEEP_LOG_HISTORY_VALIDATION = "history_validation"

DEEP_LOGS = {DEEP_LOG_NOTIFICATION: [get_main_path(), DEEP_EXT_LOGS],
             DEEP_LOG_HISTORY_TRAIN_BATCHES: [get_main_path(), DEEP_EXT_CSV],
             DEEP_LOG_HISTORY_TRAIN_EPOCHS: [get_main_path(), DEEP_EXT_CSV],
             DEEP_LOG_HISTORY_VALIDATION: [get_main_path(), DEEP_EXT_CSV]}
