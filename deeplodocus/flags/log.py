from deeplodocus.flags import Flag
from deeplodocus.flags.ext import DEEP_EXT_CSV, DEEP_EXT_LOGS
from deeplodocus.utils import get_main_path

DEEP_LOG_NOTIFICATION = "notification"
DEEP_HISTORY_TRAIN_BATCHES = Flag(
    name="Training Batches History",
    description="History : history of training batches",
    names=["train_batches", "train batches", "training batches", "training_batches"]
)

DEEP_HISTORY_TRAIN_EPOCHS = Flag(
    name="Training Epochs History",
    description="History : history of training epochs",
    names=["train_epochs", "train epochs", "training epochs", "training_epochs"]
)

DEEP_HISTORY_VALIDATION = Flag(
    name="Validation History",
    description="History : validation history",
    names=["train_epochs", "train epochs", "training epochs", "training_epochs"]
)

DEEP_LOGS = {DEEP_LOG_NOTIFICATION: [get_main_path(), DEEP_EXT_LOGS],
             DEEP_HISTORY_TRAIN_BATCHES: [get_main_path(), DEEP_EXT_CSV],
             DEEP_HISTORY_TRAIN_EPOCHS: [get_main_path(), DEEP_EXT_CSV],
             DEEP_HISTORY_VALIDATION: [get_main_path(), DEEP_EXT_CSV]}


DEEP_LOG_RESULT_DIRECTORIES = ["logs", "weights", "history"]

