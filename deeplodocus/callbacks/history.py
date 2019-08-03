# Python imports
import pandas as pd
import time
import datetime
from typing import Union
import multiprocessing.managers
import copy
import os

# Deeplodocus imports
from deeplodocus.utils.notification import Notification
from deeplodocus.utils.dict_utils import merge_sum_dict
from deeplodocus.core.metrics.over_watch_metric import OverWatchMetric
from deeplodocus.utils.logs import Logs
from deeplodocus.brain.thalamus import Thalamus
from deeplodocus.brain.signal import Signal
from deeplodocus.utils.generic_utils import get_corresponding_flag

# Deeplodocus flags
from deeplodocus.flags import *

Num = Union[int, float]


class History(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    The class stores and manages the history
    """

    def __init__(
        self,
        metrics: dict,
        losses: dict,
        log_dir: str = "history",
        train_batches_filename: str = "history_batches_training.csv",
        train_epochs_filename: str = "history_epochs_training.csv",
        validation_filename: str = "history_validation.csv",
        verbose: Flag = DEEP_VERBOSE_BATCH,
        memorize: Flag = DEEP_MEMORIZE_BATCHES,
        save_signal: Flag = DEEP_SAVE_SIGNAL_END_EPOCH,
        overwatch_metric: OverWatchMetric = OverWatchMetric(
            name=TOTAL_LOSS,
            condition=DEEP_SAVE_CONDITION_LESS
        )
    ):

        self.log_dir = log_dir
        self.verbose = verbose
        self.metrics = metrics
        self.losses = losses
        self.memorize = get_corresponding_flag([DEEP_MEMORIZE_BATCHES, DEEP_MEMORIZE_EPOCHS], info=memorize)
        self.save_signal = save_signal
        self.overwatch_metric = overwatch_metric

        # Running metrics
        self.running_total_loss = 0
        self.running_losses = {}
        self.running_metrics = {}

        self.train_batches_history = multiprocessing.Manager().Queue()
        self.train_epochs_history = multiprocessing.Manager().Queue()
        self.validation_history = multiprocessing.Manager().Queue()

        # Add headers to history files
        train_batches_headers = ",".join([WALL_TIME, RELATIVE_TIME, EPOCH, BATCH, TOTAL_LOSS]
                                         + list(vars(losses).keys())
                                         + list(vars(metrics).keys()))
        train_epochs_headers = ",".join([WALL_TIME, RELATIVE_TIME, EPOCH,  TOTAL_LOSS]
                                        + list(vars(losses).keys())
                                        + list(vars(metrics).keys()))
        validation_headers = ",".join([WALL_TIME, RELATIVE_TIME, EPOCH,  TOTAL_LOSS]
                                      + list(vars(losses).keys())
                                      + list(vars(metrics).keys()))

        # Create the history files
        self.__add_logs("history_train_batches", log_dir, ".csv", train_batches_headers)
        self.__add_logs("history_train_epochs", log_dir, ".csv", train_epochs_headers)
        self.__add_logs("history_validation", log_dir, ".csv", validation_headers)

        self.start_time = 0
        self.paused = False

        # Filepaths
        self.log_dir = log_dir
        self.train_batches_filename = train_batches_filename
        self.train_epochs_filename = train_epochs_filename
        self.validation_filename = validation_filename

        # Load histories
        self.__load_histories()

        # Connect to signals
        Thalamus().connect(
            receiver=self.on_batch_end,
            event=DEEP_EVENT_ON_BATCH_END,
            expected_arguments=[
                "minibatch_index",
                "num_minibatches",
                "epoch_index",
                "total_loss",
                "result_losses",
                "result_metrics",
                "epoch_index"
            ]
        )
        Thalamus().connect(
            receiver=self.on_epoch_end,
            event=DEEP_EVENT_ON_EPOCH_END,
            expected_arguments=[
                "epoch_index",
                "num_epochs",
                "num_minibatches",
                "total_validation_loss",
                "result_validation_losses",
                "result_validation_metrics",
                "num_minibatches_validation"
            ]
        )
        Thalamus().connect(
            receiver=self.on_train_begin,
            event=DEEP_EVENT_ON_TRAINING_START,
            expected_arguments=[]
        )
        Thalamus().connect(
            receiver=self.on_train_end,
            event=DEEP_EVENT_ON_TRAINING_END,
            expected_arguments=[]
        )

        Thalamus().connect(
            receiver=self.on_epoch_start,
            event=DEEP_EVENT_ON_EPOCH_START,
            expected_arguments=[
                "epoch_index",
                "num_epochs"
            ]
        )
        Thalamus().connect(
            receiver=self.send_training_loss,
            event=DEEP_EVENT_REQUEST_TRAINING_LOSS,
            expected_arguments=[]
        )

    def on_train_begin(self):
        """
        AUTHORS:
        --------

        :author: Samuel Westlake
        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Called when training begins

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """

        self.__set_start_time()

    def on_epoch_start(self, epoch_index: int, num_epochs: int):
        """
        Author: Samuel Westlake
        :param epoch_index: int index of current epoch
        :param num_epochs: int: total number of epochs
        :return: None
        """

        if DEEP_VERBOSE_BATCH.corresponds(self.verbose) or DEEP_VERBOSE_EPOCH.corresponds(self.verbose):
            Notification(DEEP_NOTIF_INFO, EPOCH_START % (epoch_index, num_epochs))

    def on_batch_end(self,
                     minibatch_index: int,
                     num_minibatches: int,
                     epoch_index: int,
                     total_loss: int,
                     result_losses: dict,
                     result_metrics: dict):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Called at the end of every batch

        PARAMETERS:
        -----------

        :param minibatch_index: int: Index of the current minibatch
        :param num_minibatches: int: Number of minibatches per epoch
        :param epoch_index: int: Index of the current epoch
        :param total_loss: int: The total loss
        :param result_losses: dict: List of resulting losses
        :param result_metrics: dict: List of resulting metrics

        RETURN:
        -------

        :return: None
        """
        # Save the running metrics
        self.running_total_loss = self.running_total_loss + total_loss
        self.running_losses = merge_sum_dict(self.running_losses, result_losses)
        self.running_metrics = merge_sum_dict(self.running_metrics, result_metrics)

        # If the user wants to print stats for each batch
        if DEEP_VERBOSE_BATCH.corresponds(self.verbose):

            # Print training loss and metrics on batch end
            Thalamus().add_signal(
                Signal(
                    event=DEEP_EVENT_PRINT_TRAINING_BATCH_END,
                    args={
                        "losses": result_losses,
                        "total_loss": total_loss,
                        "metrics": result_metrics,
                        "num_minibatches": num_minibatches,
                        "minibatch_index": minibatch_index,
                        "epoch_index": epoch_index
                    }
                )
            )

        # Save the data in memory
        if DEEP_MEMORIZE_BATCHES.corresponds(self.memorize):
            # Save the history in memory
            data = [datetime.datetime.now().strftime(TIME_FORMAT),
                    self.__time(),
                    epoch_index,
                    minibatch_index,
                    total_loss] + \
                    [value.item() for (loss_name, value) in result_losses.items()] + \
                    [value for (metric_name, value) in result_metrics.items()]
            self.train_batches_history.put(data)

        # Save the history after 10 batches
        if self.train_batches_history.qsize() > 10:
            self.save(only_batches=True)

    def on_epoch_end(self,
                     epoch_index: int,
                     num_epochs: int,
                     num_minibatches: int,
                     total_validation_loss: int,
                     result_validation_losses: dict,
                     result_validation_metrics: dict,
                     num_minibatches_validation: int):
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Method for managing history at the end of each epoch

        PARAMETERS:
        -----------

        :param epoch_index: int: current epoch index
        :param num_epochs: int: total number of epoch
        :param num_minibatches: int: number of minibatches per epoch
        :param total_validation_loss:
        :param result_validation_losses:
        :param result_validation_metrics:
        :param num_minibatches_validation:

        RETURN:
        -------

        :return: None
        """
        # MANAGE TRAINING HISTORY
        if DEEP_VERBOSE_EPOCH.corresponds(self.verbose) or DEEP_VERBOSE_BATCH.corresponds(self.verbose):

            # Print the training loss and metrics on epoch end
            Thalamus().add_signal(
                Signal(
                    event=DEEP_EVENT_PRINT_TRAINING_EPOCH_END,
                    args={
                        "losses": {key: value / num_minibatches for key, value in self.running_losses.items()},
                        "total_loss": self.running_total_loss / num_minibatches,
                        "metrics": {key: value / num_minibatches for key, value in self.running_metrics.items()},
                    }
                )
            )

        # If recording on batch or epoch
        if DEEP_MEMORIZE_BATCHES.corresponds(self.memorize) or DEEP_MEMORIZE_EPOCHS.corresponds(self.memorize):
            data = [
                datetime.datetime.now().strftime(TIME_FORMAT),
                self.__time(),
                epoch_index,
                self.running_total_loss / num_minibatches
            ]\
                   + [value.item() / num_minibatches for (loss_name, value) in self.running_losses.items()]\
                   + [value / num_minibatches for (metric_name, value) in self.running_metrics.items()]
            self.train_epochs_history.put(data)

        self.running_total_loss = 0
        self.running_losses = {}
        self.running_metrics = {}

        # MANAGE VALIDATION HISTORY
        if total_validation_loss is not None:
            if DEEP_VERBOSE_EPOCH.corresponds(self.verbose) or DEEP_VERBOSE_BATCH.corresponds(self.verbose):

                # Print the validation loss and metrics on epoch end
                Thalamus().add_signal(
                    Signal(
                        event=DEEP_EVENT_PRINT_VALIDATION_EPOCH_END,
                        args={
                            "losses": result_validation_losses,
                            "total_loss": total_validation_loss,
                            "metrics": result_validation_metrics,
                        }
                    )
                )

            if DEEP_MEMORIZE_BATCHES.corresponds(self.memorize) or DEEP_MEMORIZE_EPOCHS.corresponds(self.memorize):
                data = [
                    datetime.datetime.now().strftime(TIME_FORMAT),
                    self.__time(),
                    epoch_index,
                    total_validation_loss
                ] \
                    + [value.item() for (loss_name, value) in result_validation_losses.items()] \
                    + [value for (metric_name, value) in result_validation_metrics.items()]
                self.validation_history.put(data)

        if DEEP_SAVE_SIGNAL_AUTO.corresponds(self.save_signal):
            self.__compute_overwatch_metric(
                num_minibatches_training=num_minibatches,
                running_total_loss=self.running_total_loss,
                running_losses=self.running_losses,
                running_metrics=self.running_metrics,
                total_validation_loss=total_validation_loss,
                result_validation_losses=result_validation_losses,
                result_validation_metrics=result_validation_metrics
            )
        elif DEEP_SAVE_SIGNAL_END_EPOCH.corresponds(self.save_signal):
            Thalamus().add_signal(
                Signal(
                    event=DEEP_EVENT_SAVE_MODEL,
                    args={}
                )
            )

        Notification(DEEP_NOTIF_SUCCESS, EPOCH_END % (epoch_index, num_epochs))
        self.save()

    def on_train_end(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Actions to perform when the training finishes

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        Notification(DEEP_NOTIF_SUCCESS, HISTORY_SAVED % self.log_dir)

    def save(self, only_batches=False):

        for i in range(self.train_batches_history.qsize()):
            train_batch_history = ",".join(str(value) for value in self.train_batches_history.get())
            self.__add_logs("history_train_batches", self.log_dir, DEEP_EXT_CSV, train_batch_history)

        if only_batches is False:
            for i in range(self.train_epochs_history.qsize()):
                train_epochs_history = ",".join(str(value) for value in self.train_epochs_history.get())
                self.__add_logs("history_train_epochs", self.log_dir, DEEP_EXT_CSV, train_epochs_history)

            for i in range(self.validation_history.qsize()):
                validation_history = ",".join(str(value) for value in self.validation_history.get())
                self.__add_logs("history_validation", self.log_dir, DEEP_EXT_CSV, validation_history)

    def __load_histories(self):
        """
        AUTHORS:
        --------

        :author: Samuel Westlake
        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Load possible existing histories in memory

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        # Load train batches history
        if os.path.isfile(self.__get_path(self.train_batches_filename)):
            self.train_batches_history = pd.read_csv(self.__get_path(self.train_batches_filename))
        # Load train epochs history
        if os.path.isfile(self.__get_path(self.train_epochs_filename)):
            self.train_epochs_history = pd.read_csv(self.__get_path(self.train_epochs_filename))
        # Load Validation history
        if os.path.isfile(self.__get_path(self.validation_filename)):
            self.validation_history = pd.read_csv(self.__get_path(self.validation_filename))

    def __get_path(self, file_name):
        """
        :param file_name: str: name of a file to get a path for
        :return: str: path to file name in log dir
        """
        return "%s/%s" % (self.log_dir, file_name)

    def __set_start_time(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Set the start time

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        # If we did not train the model before we start the time at NOW
        if self.train_epochs_history.empty:
            self.start_time = time.time()
        # Else use the last time of the history
        else:
            self.start_time = time.time() - self.train_epochs_history[RELATIVE_TIME][self.train_epochs_history.index[-1]]

    def __time(self):
        """
        AUTHORS:
        --------

        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Calculate the relative time between the start of the training and the current time

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return->float: Current train time in seconds
        """
        return round(time.time() - self.start_time, 2)

    def pause(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Pause the timer in order to keep the relative time coherent if restarting the training

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        self.paused = True

    @staticmethod
    def __add_logs(log_type: str, log_folder: str, log_extension: str, message: str)->None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Add the history to the corresponding log file

        PARAMETERS:
        -----------

        :param log_type->str: The type of log
        :param log_folder->str: The folder in which the log will be saved
        :param log_extension->str: The extension of the log file
        :param message->str: The message which has to be written in the log file

        RETURN:
        -------

        :return: None
        """
        Logs(log_type, log_folder, log_extension).add(message, write_time=False)

    def __compute_overwatch_metric(
            self,
            num_minibatches_training,
            running_total_loss,
            running_losses,
            running_metrics,
            total_validation_loss,
            result_validation_losses,
            result_validation_metrics) -> None:
        """
        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Compute the overwatch metric and send it to the saver

        PARAMETERS:
        -----------

        :param num_minibatches_training:
        :param running_total_loss:
        :param running_losses:
        :param running_metrics:
        :param total_validation_loss:
        :param result_validation_losses:
        :param result_validation_metrics:


        RETURN:
        -------

        :return:
        """

        # If the validation loss is None (No validation) we take the metric from the training as overwatch metric
        if total_validation_loss is None:
            data = dict([(TOTAL_LOSS, running_total_loss / num_minibatches_training)]
                        + [(loss_name, value.item() / num_minibatches_training) for (loss_name, value) in running_losses.items()]
                        + [(metric_name, value / num_minibatches_training) for (metric_name, value) in running_metrics.items()])

            for key, value in data.items():
                if key == self.overwatch_metric.get_name():
                    self.overwatch_metric.set_value(value)
                    break
        else:
            data = dict([(TOTAL_LOSS, total_validation_loss)] +
                        [(loss_name, value.item()) for (loss_name, value) in result_validation_losses.items()]
                        + [(metric_name, value / num_minibatches_training) for (metric_name, value) in result_validation_metrics.items()])

            for key, value in data.items():
                if key == self.overwatch_metric.get_name():
                    self.overwatch_metric.set_value(value)
                    break

        Thalamus().add_signal(
            Signal(
                event=DEEP_EVENT_OVERWATCH_METRIC_COMPUTED,
                args={"current_overwatch_metric": copy.deepcopy(self.overwatch_metric)}
            )
        )

    def get_overwatch_metric(self) -> OverWatchMetric:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get a deep copy of the over watched metric

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return (OverwatchMetric) : A deep copy of the over watched metric
        """

        # Always return a deep copy of the over watch metric to avoid any issue
        return copy.deepcopy(self.overwatch_metric)

    def send_training_loss(self):
        Thalamus().add_signal(
            Signal(
                event=DEEP_EVENT_SEND_TRAINING_LOSS,
                args={"training_loss": self.running_total_loss}
            )
        )
