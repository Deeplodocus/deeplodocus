# Python import
from typing import Union

# Deeplodocus imports
from deeplodocus.utils.flag import Flag
from deeplodocus.utils.generic_utils import get_corresponding_flag

# Deeplodocus flags
from deeplodocus.utils.flags.compare_metric import *
from deeplodocus.utils.flags.flag_lists import DEEP_LIST_COMPARE_METRIC
from deeplodocus.utils.flags import TOTAL_LOSS


class OverWatchMetric(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Metric to overwatch during the training
    """

    def __init__(self, name: str = TOTAL_LOSS, condition: Union[Flag, int, str, None] = DEEP_COMPARE_METRIC_SMALLER):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize the OverWatchMetric instance

        PARAMETERS:
        -----------
        :param name (str): The name of the metric to over watch
        :param condition (Flag):
        """
        self.name = name
        self.value = 0.0
        self.condition = get_corresponding_flag(flag_list=DEEP_LIST_COMPARE_METRIC, info=condition, fatal=False, default=DEEP_COMPARE_METRIC_SMALLER)

    """
    "
    " SETTERS
    "
    """
    def set_value(self, value: float):
        self.value = value

    """
    "
    " GETTERS
    "
    """
    def get_name(self):
        return self.name

    def get_value(self):
        return self.value

    def get_condition(self):
        return self.condition
