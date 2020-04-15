import os
import copy
import time
import datetime
from decimal import Decimal
from typing import Union

# Deeplodocus imports
from deeplodocus.brain.signal import Signal
from deeplodocus.brain.thalamus import Thalamus
from deeplodocus.core.metrics import OverWatch
from deeplodocus.flags import *
from deeplodocus.utils.logs import Logs
from deeplodocus.utils.notification import Notification
from deeplodocus.utils.generic_utils import get_corresponding_flag

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
        save_signal: Flag = DEEP_SAVE_SIGNAL_END_EPOCH,
        overwatch_metric: OverWatch = OverWatch(
            name=DEEP_LOG_TOTAL_LOSS,
            condition=DEEP_SAVE_CONDITION_LESS
        ),
        write_interval: int = 10,
        enable_train_batches: bool = True,
        enable_train_epochs: bool = True,
        enable_validation: bool = True,
        overwrite: bool = None
    ):
        self.log_dir = log_dir
        self.verbose = get_corresponding_flag(DEEP_LIST_VERBOSE, verbose)
        self.save_signal = get_corresponding_flag(DEEP_LIST_SAVE_SIGNAL, save_signal)
        self.overwatch_metric = overwatch_metric
        self.write_interval = write_interval
        self.overwrite = overwrite
        self.file_paths = {
            flag.var_name: "/".join((log_dir, file_name))
            for flag, file_name in zip(
                DEEP_LIST_LOG_HISTORY,
                (train_batches_filename, train_epochs_filename, validation_filename)
            )
        }
        self.enabled = {
            flag.var_name: enabled
            for flag, enabled in zip(
                DEEP_LIST_LOG_HISTORY,
                (enable_train_batches, enable_train_epochs, enable_validation)
            )
        }
        self.headers = {
            DEEP_LOG_TRAIN_BATCHES.var_name: [
                flag.name for flag in DEEP_LIST_HISTORY_HEADER
            ],
            DEEP_LOG_TRAIN_EPOCHS.var_name: [
                flag.name for flag in DEEP_LIST_HISTORY_HEADER if not flag.corresponds(DEEP_LOG_BATCH)
            ],
            DEEP_LOG_VALIDATION.var_name: [
                flag.name for flag in DEEP_LIST_HISTORY_HEADER if not flag.corresponds(DEEP_LOG_BATCH)
            ],
        }
        self._training_start = None
        self._batch_data = {}
        self._loss_data = {item: None for item in (TRAINING, VALIDATION)}
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
        self.__init_files()

    def __init_files(self):
        # Check if each of the expected history files already exist
        exists = [
            True if os.path.exists(file_path) and os.path.getsize(file_path) > 0 else False
            for file_name, file_path in self.file_paths.items()
        ]
        # If history files exist and we don't know if they can be overwritten or not
        if self.overwrite is None and any(exists):
            # Inform user what exists
            Notification(DEEP_NOTIF_WARNING, "The following history files already exist : ")
            for ex, (file_name, path) in zip(exists, self.file_paths.items()):
                if ex:
                    Notification(DEEP_NOTIF_WARNING, "\t- %s : %s" % (file_name, path))
            # Ask if they can be overwritten or not
            while self.overwrite is None:
                r = Notification(DEEP_NOTIF_INPUT, "Would you like to overwrite them? (y/n)").get()
                if DEEP_RESPONSE_YES.corresponds(r):
                    self.overwrite = True
                    break
                elif DEEP_RESPONSE_NO.corresponds(r):
                    self.overwrite = False
                    break
        for en, (_, ex), (file_name, path) in zip(exists, self.enabled.items(), self.file_paths.items()):
            # If file exists and is enabled
            if en and (not ex or self.overwrite):
                with open(path, "w"):
                    pass
            elif self.overwrite and ex and not en:
                os.remove(path)

    def __init_file(self, file_name):
        # If a file is found to not exists, use this to re-initialise it
        with open(self.file_paths[file_name], "w") as file:
            file.write("%s\n" % ",".join(self.headers[file_name]))

    def __update_header(self, file_name):
        # If the self.header changes, use this to update the header in the file
        with open(self.file_paths[file_name], "r") as file:
            lines = file.readlines()
        lines[0] = "%s\n" % ",".join(self.headers[file_name])
        with open(self.file_paths[file_name], "w") as file:
            file.writelines(lines)

    def __write_lines(self, file_name, data):
        # Check all given keys
        for key in data.keys():
            update_header = False
            # If key is not in hte file header
            if key not in self.headers[file_name]:
                self.headers[file_name].append(key)
                update_header = True
            if update_header:
                # Update file header
                self.__update_header(file_name)
        lines = []
        for i in range(len(data[list(data.keys())[0]])):
            line = []
            for key in self.headers[file_name]:
                try:
                    line.append(str(data[key][i]))
                except KeyError:
                    line.append("")
            lines.append(",".join(line))
        with open(self.file_paths[file_name], "a") as file:
            file.write("%s\n" % "\n".join(lines))
        self._batch_data = {}

    def __write_line(self, file_name, data):
        # Check all given keys
        for key in data.keys():
            update_header = False
            # If key is not in hte file header
            if key not in self.headers[file_name]:
                self.headers[file_name].append(key)
                update_header = True
            if update_header:
                # Update file header
                self.__update_header(file_name)
        line = []
        for key in self.headers[file_name]:
            try:
                line.append(str(data[key]))
            except KeyError:
                line.append("")
        with open(self.file_paths[file_name], "a") as file:
            file.write("%s\n" % ",".join(line))
        self._batch_data = {}

    def on_train_begin(self):
        # Write history file headers
        for file_name, file_path in self.file_paths.items():
            # If the file exists and cannot be overwritten
            if not self.overwrite and os.path.exists(file_path) and os.path.getsize(file_path):
                # Get the header of the file
                with open(file_path, "r") as file:
                    new_header = file.readline().strip().split(",")
                # Add any cols that are expected but missing
                for item in self.headers[file_name]:
                    if item not in new_header:
                        new_header.append(item)
                # Update header
                self.headers[file_name] = new_header
                self.__update_header(file_name)
            elif self.enabled[file_name]:
                self.__init_file(file_name)
        self._training_start = time.time()

    def on_epoch_start(self, epoch_index: int, num_epochs: int):
        if DEEP_VERBOSE_BATCH.corresponds(self.verbose) or DEEP_VERBOSE_EPOCH.corresponds(self.verbose):
            Notification(DEEP_NOTIF_INFO, EPOCH_START % (epoch_index, num_epochs))

    def on_batch_end(self, loss, losses, metrics, epoch_index, batch_index, num_batches):
        if self.verbose.corresponds(DEEP_VERBOSE_BATCH):
            Notification(
                DEEP_NOTIF_RESULT, "[%i : %i/%i] : %s" % (
                    epoch_index,
                    batch_index,
                    num_batches,
                    self.compose_text(loss, losses, metrics)
                )
            )
        if self.enabled[DEEP_LOG_TRAIN_BATCHES.var_name]:
            data = {
                DEEP_LOG_WALL_TIME.name: datetime.datetime.now().strftime(TIME_FORMAT),
                DEEP_LOG_RELATIVE_TIME.name: time.time() - self._training_start,
                DEEP_LOG_EPOCH.name: epoch_index,
                DEEP_LOG_BATCH.name: batch_index,
                DEEP_LOG_TOTAL_LOSS.name: loss,
                **losses,
                **metrics
            }
            # Update list of batch data
            for key, v in data.items():
                try:
                    self._batch_data[key].append(v)
                except KeyError:
                    self._batch_data[key] = [v]

            if not batch_index % self.write_interval:
                self.__write_lines(DEEP_LOG_TRAIN_BATCHES.var_name, self._batch_data)

    def on_epoch_end(self, epoch_index: int, loss: float, losses: dict, metrics: dict):
        self._loss_data[TRAINING] = loss
        # If some batch data remains to be written
        if self._batch_data and self.enabled[DEEP_LOG_TRAIN_BATCHES.var_name]:
            self.__write_lines(DEEP_LOG_TRAIN_BATCHES.var_name, self._batch_data)
        # If verbose is Batch or Epoch
        if DEEP_VERBOSE_EPOCH.corresponds(self.verbose) or DEEP_VERBOSE_BATCH.corresponds(self.verbose):
            Notification(
                DEEP_NOTIF_RESULT,
                "Epoch %i : %s : %s" % (epoch_index, TRAINING, self.compose_text(loss, losses, metrics))
            )
        if self.enabled[DEEP_LOG_TRAIN_EPOCHS.var_name]:
            data = {
                DEEP_LOG_WALL_TIME.name: datetime.datetime.now().strftime(TIME_FORMAT),
                DEEP_LOG_RELATIVE_TIME.name: time.time() - self._training_start,
                DEEP_LOG_EPOCH.name: epoch_index,
                DEEP_LOG_TOTAL_LOSS.name: loss,
                **losses,
                **metrics
            }
            self.__write_line(DEEP_LOG_TRAIN_EPOCHS.var_name, data)

    def on_validation_end(self, epoch_index: int, loss: float, losses: dict, metrics: dict):
        self._loss_data[VALIDATION] = loss
        # If verbose is Batch or Epoch
        if DEEP_VERBOSE_EPOCH.corresponds(self.verbose) or DEEP_VERBOSE_BATCH.corresponds(self.verbose):
            Notification(
                DEEP_NOTIF_RESULT,
                "Epoch %i : %s : %s" % (epoch_index, VALIDATION, self.compose_text(loss, losses, metrics))
            )
        if self.enabled[DEEP_LOG_VALIDATION.var_name]:
            data = {
                DEEP_LOG_WALL_TIME.name: datetime.datetime.now().strftime(TIME_FORMAT),
                DEEP_LOG_RELATIVE_TIME.name: time.time() - self._training_start,
                DEEP_LOG_EPOCH.name: epoch_index,
                DEEP_LOG_TOTAL_LOSS.name: loss,
                **losses,
                **metrics
            }
            self.__write_line(DEEP_LOG_VALIDATION.var_name, data)

    def on_train_end(self):
        pass

    def send_training_loss(self):
        Thalamus().add_signal(
            Signal(
                event=DEEP_EVENT_SEND_TRAINING_LOSS,
                args={DEEP_LOG_VALIDATION.var_name: self._loss_data[DEEP_LOG_VALIDATION.var_name]}
            )
        )

    @staticmethod
    def compose_text(total_loss, losses, metrics, sep=" : "):
        return sep.join(
            ["%s : %.4e" % (DEEP_LOG_TOTAL_LOSS.name, Decimal(total_loss))]
            + ["%s : %.4e" % (loss_name, Decimal(value)) for (loss_name, value) in losses.items()]
            + ["%s : %.4e " % (metric_name, Decimal(value)) for (metric_name, value) in metrics.items()]
        )
