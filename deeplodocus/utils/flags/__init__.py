from deeplodocus.utils.flags.cmd import *
from deeplodocus.utils.flags.config import *
from deeplodocus.utils.flags.entry import *
from deeplodocus.utils.flags.exit import *
from deeplodocus.utils.flags.ext import *
from deeplodocus.utils.flags.filter import *
from deeplodocus.utils.flags.lib import *
from deeplodocus.utils.flags.log import *
from deeplodocus.utils.flags.msg import *
from deeplodocus.utils.flags.notif import *
from deeplodocus.utils.flags.path import *
from deeplodocus.utils.flags.dtype import *
from deeplodocus.utils.flags.ui import *

#
# MODEL SAVING CONDITION
#
DEEP_SAVE_CONDITION_END_BATCH = 0         # Highly not recommended
DEEP_SAVE_CONDITION_END_EPOCH = 1
DEEP_SAVE_CONDITION_END_TRAINING = 2
DEEP_SAVE_CONDITION_AUTO = 3                # Highly recommended

#
# DATA MEMORIZATION CONDITION
#

DEEP_MEMORIZE_BATCHES = 0
DEEP_MEMORIZE_EPOCHS = 1


#
# VERBOSE
#
DEEP_VERBOSE_BATCH = 2
DEEP_VERBOSE_EPOCH = 1
DEEP_VERBOSE_TRAINING = 0

#
# SHUFFLE
#
DEEP_SHUFFLE_NONE = 0
DEEP_SHUFFLE_BATCHES = 1
DEEP_SHUFFLE_ALL = 2

#
# SAVE NETWORK FORMAT
#
DEEP_SAVE_NET_FORMAT_ONNX = 0
DEEP_SAVE_NET_FORMAT_PYTORCH = 1

#
# COMPARISON FOR THE OVERWATCH METRIC
#
DEEP_COMPARE_SMALLER = 0
DEEP_COMPARE_BIGGER = 1

#
# DEEP_ENCODE_FLAGS
#
DEEP_ENCODE_ASCII = "ascii"
DEEP_ENCODE_UTF8 = "utf-8"

FINISHED_TRAINING = "Finished training"
SUMMARY = "Summary"
TOTAL_LOSS = "Total Loss"
WALL_TIME = "Wall Time"
RELATIVE_TIME = "Relative Time"
EPOCH = "Epoch"
BATCH = "Batch"
TRAINING = "Training"
VALIDATION = "Validation"
TIME_FORMAT = "%Y:%m:%d:%H:%M:%S"
EPOCH_END = "End of Epoch [%i/%i]"
EPOCH_START = "Start of Epoch [%i/%i]"
HISTORY_SAVED = "History saved to %s"
