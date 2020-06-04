from deeplodocus.utils.generic_utils import get_corresponding_flag
from deeplodocus.utils.notification import Notification
from deeplodocus.flags import *

from typing import Union


class OverWatch(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Metric to overwatch during the training
    """

    def __init__(
            self,
            metric: str = DEEP_LOG_TOTAL_LOSS,
            condition: Union[Flag, None] = DEEP_SAVE_CONDITION_LESS,
            dataset: Union[Flag, None] = DEEP_DATASET_VAL
    ):
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
        self.metric = metric
        self.dataset = DEEP_DATASET_VAL if dataset is None \
            else get_corresponding_flag([DEEP_DATASET_TRAIN, DEEP_DATASET_VAL], dataset)
        self.current_best = None
        self._condition = get_corresponding_flag(DEEP_LIST_SAVE_CONDITIONS, condition)
        self._is_better = None
        self.set_is_better()

    def watch(self, dataset: Flag, loss, losses, metrics=None):
        if self.dataset.corresponds(dataset):
            value = {**losses, **metrics, DEEP_LOG_TOTAL_LOSS.name: loss}[self.metric]
            if self.current_best is None:
                self.current_best = value
                return True
            elif self._is_better(value):
                Notification(
                    DEEP_NOTIF_SUCCESS,
                    "%s improved from %.4e to %.4e : Improvement of %.2f" %
                    (self.metric, self.current_best, value, self.percent(value)) + "%"
                )
                self.current_best = value
                return True
            Notification(DEEP_NOTIF_INFO, "No improvement")
        return False

    @property
    def condition(self):
        return self._condition

    @condition.setter
    def condition(self, condition):
        self._condition = condition
        self.set_is_better()

    def is_greater(self, x):
        if x >= self.current_best:
            return True
        else:
            return False

    def is_less(self, x):
        if x <= self.current_best:
            return True
        else:
            return False

    def set_is_better(self):
        if self.condition.corresponds(DEEP_SAVE_CONDITION_LESS):
            self._is_better = self.is_less
        elif self.condition.corresponds(DEEP_SAVE_CONDITION_GREATER):
            self._is_better = self.is_greater
        else:
            Notification(DEEP_NOTIF_FATAL, "OverWatch : Unknown condition : " % self.condition)

    def percent(self, x):
        return abs(self.current_best - x) / self.current_best * 100
