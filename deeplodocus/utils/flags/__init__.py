import os
import __main__

from deeplodocus.utils.flags.ext import *
from deeplodocus.utils.flags.config import *
from deeplodocus.utils.flags.msg import *
from deeplodocus.utils.flags.notif import *
from deeplodocus.utils.flags.entry import *
from deeplodocus.utils.flags.lib import *
from deeplodocus.utils.flags.filter import *

#
# HISTORY SAVING CONDITION
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
DEEP_SHUFFLE_ALL = 1
DEEP_SHUFFLE_BATCHES = 2

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
# ABSOLUTE PATHS TO WORKING DIRECTORIES
#
DEEP_PATH_NOTIFICATION = r"%s/logs" % os.path.dirname(os.path.abspath(__main__.__file__))
DEEP_PATH_RESULTS = r"%s/results" % os.path.dirname(os.path.abspath(__main__.__file__))
DEEP_PATH_HISTORY = r"%s/results/history" % os.path.dirname(os.path.abspath(__main__.__file__))
DEEP_PATH_SAVE_MODEL = r"%s/results/models" % os.path.dirname(os.path.abspath(__main__.__file__))

#
# DEEP_ENCODE_FLAGS
#
DEEP_ENCODE_ASCII = "ascii"
DEEP_ENCODE_UTF8 = "utf-8"

#
# DEEP_EXIT_FLAGS
#
DEEP_EXIT_FLAGS = ["q", "quit", "exit"]

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
