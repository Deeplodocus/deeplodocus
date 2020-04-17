from deeplodocus.flags import Flag
from deeplodocus.flags.ext import DEEP_EXT_CSV, DEEP_EXT_LOGS
from deeplodocus.utils import get_main_path

DEEP_LOG_NOTIFICATION = "notification"

DEEP_LOG_TRAIN_BATCHES = Flag(
    name="Training Batches History",
    description="History : history of training batches",
    names=["train_batches", "train batches", "training batches", "training_batches"]
)
DEEP_LOG_TRAIN_EPOCHS = Flag(
    name="Training Epochs History",
    description="History : history of training epochs",
    names=["train_epochs", "train epochs", "training epochs", "training_epochs"]
)
DEEP_LOG_VALIDATION = Flag(
    name="Validation History",
    description="History : validation history",
    names=["train_epochs", "train epochs", "training epochs", "training_epochs"]
)

DEEP_LOG_WALL_TIME = Flag(
    name="Wall Time",
    description="History : Wall Time header",
    names=["wall time", "wall_time"]

)
DEEP_LOG_RELATIVE_TIME = Flag(
    name="Relative Time",
    description="History : Relative Time header",
    names=["relative time", "relative_time"]

)
DEEP_LOG_EPOCH = Flag(
    name="Epoch",
    description="History : Epoch header",
    names=["epoch"]

)
DEEP_LOG_BATCH = Flag(
    name="Batch",
    description="History : Batch header",
    names=["batch"]

)
DEEP_LOG_TOTAL_LOSS = Flag(
    name="Total Loss",
    description="History : Total Loss header",
    names=["total loss", "total_loss"]

)
DEEP_LOGS = {
    DEEP_LOG_NOTIFICATION: [get_main_path(), DEEP_EXT_LOGS],
    DEEP_LOG_TRAIN_BATCHES: [get_main_path(), DEEP_EXT_CSV],
    DEEP_LOG_TRAIN_EPOCHS: [get_main_path(), DEEP_EXT_CSV],
    DEEP_LOG_VALIDATION: [get_main_path(), DEEP_EXT_CSV]
}

DEEP_LOG_RESULT_DIRECTORIES = ["logs", "weights", "history"]

FINISHED_TRAINING = "Finished training"
SUMMARY = "Summary"
TRAINING = "Training"
VALIDATION = "Validation"
TIME_FORMAT = "%Y:%m:%d:%H:%M:%S"
EPOCH_END = "End of Epoch [%i/%i]"
EPOCH_START = "Start of Epoch [%i/%i]"
HISTORY_SAVED = "History saved to %s"
