from deeplodocus.utils import get_main_path
from deeplodocus.utils.flag import Flag
#
# ENTRIES
#
DEEP_ENTRY_INPUT = 0
DEEP_ENTRY_LABEL = 1
DEEP_ENTRY_OUTPUT = 2
DEEP_ENTRY_ADDITIONAL_DATA = 3


DEEP_ENTRY_INPUT_ = Flag(description="Input entry", names = ["input", "inputs", "inp", "x"])
DEEP_ENTRY_LABEL_ = Flag(description="Label entry", names = ["label", "labels", "lab", "expected output", "expected_output", "y_expected", "y_hat"])
DEEP_ENTRY_OUTPUT_ = Flag(description="Output entry", names=["output", "outputs", "y"])
DEEP_ENTRY_ADDITIONAL_DATA_ = Flag(description="Additional Data entry", names=["additional_data", "additional data"])

DEEP_ENTRY_BASE_FILE_NAME = get_main_path() + "data/auto-generated_entry_%s_%i.dat"