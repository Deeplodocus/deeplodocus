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
        log_dir: str = "history",
        train_batches_filename: str = "history_train_batches.csv",
        train_epochs_filename: str = "history_train_epochs.csv",
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
        self.memorize = get_corresponding_flag([DEEP_MEMORIZE_BATCHES, DEEP_MEMORIZE_EPOCHS], info=memorize)
        self.save_signal = save_signal
        self.overwatch_metric = overwatch_metric
        self.file_paths = {
            "train_batches": "/".join((log_dir, train_batches_filename)),
            "train_epochs": "/".join((log_dir, train_epochs_filename)),
            "validation": "/".join((log_dir, validation_filename))
        }
        self.header = {[WALL_TIME, RELATIVE_TIME, EPOCH, BATCH, TOTAL_LOSS]

        self.__init_files()

        # Connect to signals
        Thalamus().connect(
            receiver=self.on_batch_end,
            event=DEEP_EVENT_ON_BATCH_END,
            expected_arguments=["batch_index", "num_batches", "epoch_index", "loss", "losses", "metrics"]
        )
        Thalamus().connect(
            receiver=self.on_epoch_end,
            event=DEEP_EVENT_ON_EPOCH_END,
            expected_arguments=["epoch_index", "loss", "losses",  "metrics",]
        )
        Thalamus().connect(
            receiver=self.on_validation_end,
            event=DEEP_EVENT_ON_VALIDATION_END,
            expected_arguments=["epoch_index", "loss", "losses", "metrics"]
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
            expected_arguments=["epoch_index", "num_epochs"]
        )
        Thalamus().connect(
            receiver=self.send_training_loss,
            event=DEEP_EVENT_REQUEST_TRAINING_LOSS,
            expected_arguments=[]
        )

    def __init_files(self):
        # Check if each history file exists
        # (A file is considered to not exists if it is empty)
        exists = {
            file_name: {
                "exists": os.path.exists(file_path) and os.path.getsize(file_path) > 0,
                "file_path": file_path
            } for file_name, file_path in self.file_paths.items()
        }

        # Inform user of already existing history files
        # Ask if they may be overwritten - if not, they will be appended to
        overwrite = False
        if any([v["exists"] for _, v in exists.items()]):
            Notification(DEEP_NOTIF_WARNING, "The following history files already exist : ")
            for file_name, v in exists.items():
                if v["exists"]:
                    Notification(DEEP_NOTIF_WARNING, "\t- %s : %s" % (file_name, v["file_path"]))
            response = Notification(DEEP_NOTIF_INPUT, "Would you like to overwrite them? (y/n)").get()
            if get_corresponding_flag(DEEP_LIST_RESPONSE, response).corresponds(DEEP_RESPONSE_YES):
                overwrite = True

        for file_name, v in exists.items():
            if overwrite or not[v["exists"]]:
                with open(v["file_path"], "w") as _:
                    pass

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

    def on_batch_end(self, batch_index: int, num_batches: int, epoch_index: int, loss: float, losses: dict, metrics: dict):
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
        :param total_loss:
        :param losses:
        :param metrics:

        RETURN:
        -------

        :return: None
        """
        # If the user wants to print stats for each batch
        if DEEP_VERBOSE_BATCH.corresponds(self.verbose):
            # Print training loss and metrics on batch end
            Thalamus().add_signal(
                Signal(
                    event=DEEP_EVENT_PRINT_TRAINING_BATCH_END,
                    args={
                        "loss": loss,
                        "losses": losses,
                        "metrics": metrics,
                        "num_batches": num_batches,
                        "batch_index": batch_index,
                        "epoch_index": epoch_index
                    }
                )
            )
        return

        # Save the data in memory
        if DEEP_MEMORIZE_BATCHES.corresponds(self.memorize):
            # Save the history in memory
            data = [datetime.datetime.now().strftime(TIME_FORMAT),
                    self.__time(),
                    epoch_index,
                    minibatch_index,
                    total_loss] + \
                    [value.item() for (loss_name, value) in losses.items()] + \
                    [value for (metric_name, value) in metrics.items()]
            self.train_batches_history.put(data)

        # Save the history after 10 batches
        if self.train_batches_history.qsize() > 10:
            self.save(only_batches=True)

    def on_epoch_end(self, epoch_index: int, loss: float, losses: dict, metrics: dict):
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
                    args={"epoch_index": epoch_index, "loss": loss, "losses": losses, "metrics": metrics}
                )
            )
        return

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

    def on_validation_end(self, epoch_index: int, loss: float, losses: dict, metrics: dict):
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


        RETURN:
        -------

        :return: None
        """
        # MANAGE TRAINING HISTORY
        if DEEP_VERBOSE_EPOCH.corresponds(self.verbose) or DEEP_VERBOSE_BATCH.corresponds(self.verbose):

            # Print the training loss and metrics on epoch end
            Thalamus().add_signal(
                Signal(
                    event=DEEP_EVENT_PRINT_VALIDATION_EPOCH_END,
                    args={"epoch_index": epoch_index, "loss": loss, "losses": losses, "metrics": metrics}
                )
            )
        return

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
