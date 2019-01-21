from deeplodocus.utils.flags.dtype import *
from deeplodocus.utils.flags.source import *
from deeplodocus.utils.flags.load import *
from deeplodocus.utils.flags.entry import *
from deeplodocus.utils.flags.lib import *
from deeplodocus.utils.flags.transformer import *
from deeplodocus.utils.flags.shuffle import *
from deeplodocus.utils.flags.save import *
from deeplodocus.utils.flags.verbose import *


# DATA TYPES
DEEP_LIST_DTYPE = [DEEP_DTYPE_IMAGE,
                   DEEP_DTYPE_VIDEO,
                   DEEP_DTYPE_BOOLEAN,
                   DEEP_DTYPE_INTEGER,
                   DEEP_DTYPE_FLOAT,
                   DEEP_DTYPE_SEQUENCE,
                   DEEP_DTYPE_AUDIO,
                   DEEP_DTYPE_NP_ARRAY]

# SOURCES
DEEP_LIST_SOURCE = [DEEP_SOURCE_FILE,
                    DEEP_SOURCE_FOLDER,
                    DEEP_SOURCE_DATABASE]

# LOAD METHODS
DEEP_LIST_LOAD_METHOD = [DEEP_LOAD_METHOD_MEMORY,
                         DEEP_LOAD_METHOD_HARDDRIVE,
                         DEEP_LOAD_METHOD_SERVER]
# ENTRIES
DEEP_LIST_ENTRY = [DEEP_ENTRY_INPUT,
                   DEEP_ENTRY_LABEL,
                   DEEP_ENTRY_OUTPUT,
                   DEEP_ENTRY_ADDITIONAL_DATA]

DEEP_LIST_POINTER_ENTRY = [DEEP_ENTRY_INPUT,
                           DEEP_ENTRY_LABEL,
                           DEEP_ENTRY_ADDITIONAL_DATA]

# COMPUTER VISION LIBRARIES
DEEP_LIST_CV_LIB = [DEEP_LIB_PIL,
                    DEEP_LIB_OPENCV]


# TRANSFORMERS
DEEP_LIST_TRANSFORMERS = [DEEP_TRANSFORMER_SEQUENTIAL,
                          DEEP_TRANSFORMER_ONE_OF,
                          DEEP_TRANSFORMER_SOME_OF]

# SHUFFLING
DEEP_LIST_SHUFFLE = [DEEP_SHUFFLE_NONE,
                     DEEP_SHUFFLE_BATCHES,
                     DEEP_SHUFFLE_ALL]

# SAVE FORMATS
DEEP_LIST_SAVE_FORMATS = [DEEP_SAVE_FORMAT_ONNX,
                          DEEP_SAVE_FORMAT_PYTORCH]

# SAVE CONDITIONS
DEEP_LIST_SAVE_CONDITIONS = [DEEP_SAVE_CONDITION_LESS,
                             DEEP_SAVE_CONDITION_GREATER]

DEEP_LIST_VERBOSE = [DEEP_VERBOSE_BATCH,
                     DEEP_VERBOSE_EPOCH,
                     DEEP_VERBOSE_TRAINING]

