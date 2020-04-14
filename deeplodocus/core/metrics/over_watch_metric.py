# Python import
from typing import Union

# Deeplodocus imports
from deeplodocus.flags import *
from deeplodocus.utils.generic_utils import get_corresponding_flag


class OverWatchMetric(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Metric to overwatch during the training
    """

    def __init__(self, name: str = DEEP_LOG_TOTAL_LOSS, condition: Union[Flag, int, str, None] = DEEP_SAVE_CONDITION_LESS):
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
        self.condition = get_corresponding_flag(
            flag_list=DEEP_LIST_SAVE_CONDITIONS,
            info=condition,
            fatal=False,
            default=DEEP_SAVE_CONDITION_LESS
        )

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
