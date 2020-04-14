# Import modules from Python library
import datetime
from typing import Union

# Import modules from deeplodocus
from deeplodocus.callbacks.saver import Saver
from deeplodocus.callbacks.history import History
from deeplodocus.core.metrics import Metrics
from deeplodocus.core.metrics.over_watch_metric import OverWatchMetric
from deeplodocus.utils.generic_utils import generate_random_alphanumeric

# Deeplodocus flags
from deeplodocus.flags import *
from deeplodocus.utils.generic_utils import get_corresponding_flag

Num = Union[int, float]


class Hippocampus(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy
    :author: Samuel Westlake

    DESCRIPTION:
    ------------

    The Hippocampus class manages all the instances related to the information saved for short
    and long terms by Deeplodocus

    The following information are handled by the Hippocampus:
        - The history
        - The saving of the model and weights
    """

    def __init__(
            self,
            verbose: Flag = DEEP_VERBOSE_BATCH,
            overwatch_metric: OverWatchMetric = OverWatchMetric(
                name=DEEP_LOG_TOTAL_LOSS,
                condition=DEEP_SAVE_CONDITION_LESS
            ),
            save_signal: Flag = DEEP_SAVE_SIGNAL_AUTO,
            method: Flag = DEEP_SAVE_FORMAT_PYTORCH,
            enable_train_batches: bool = True,
            enable_train_epochs: bool = True,
            enable_validation: bool = True,
            overwrite: bool = False,
            history_directory: str = "history",
            weights_directory: str = "weights"
    ):
        save_signal = get_corresponding_flag(
            DEEP_LIST_SAVE_SIGNAL,
            info=save_signal,
            default=DEEP_SAVE_SIGNAL_AUTO
        )
        self.history = History(
            log_dir=history_directory,
            save_signal=save_signal,
            verbose=verbose,
            overwatch_metric=overwatch_metric,
            enable_train_batches=enable_train_batches,
            enable_train_epochs=enable_train_epochs,
            enable_validation=enable_validation
        )
        self.saver = Saver(
            save_directory=weights_directory,
            save_signal=save_signal,
            method=method,
            overwrite=overwrite
        )
