from deeplodocus.utils.flags.dtype import *
from deeplodocus.utils.flags.source import *
from deeplodocus.utils.flags.load import *
from deeplodocus.utils.flags.entry import *

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
DEEP_LIST_ENTRY = [DEEP_ENTRY_INPUT_,
                   DEEP_ENTRY_LABEL_,
                   DEEP_ENTRY_OUTPUT_,
                   DEEP_ENTRY_ADDITIONAL_DATA_]