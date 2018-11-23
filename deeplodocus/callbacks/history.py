import pandas as pd
import time
import datetime
from typing import Union
from queue import Queue
import copy
import os

from deeplodocus.utils.flags import *
from deeplodocus.utils.notification import Notification
from deeplodocus.utils.dict_utils import merge_sum_dict
from deeplodocus.core.metrics.over_watch_metric import OverWatchMetric
from deeplodocus.utils.logs import Logs
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

    def __init__(self,
                 metrics: dict,
                 losses: dict,
                 log_dir: str = DEEP_PATH_HISTORY,
                 train_batches_filename: str = "history_batches_training.csv",
                 train_epochs_filename: str = "history_epochs_training.csv",
                 validation_filename: str = "history_validation.csv",
                 verbose: int = DEEP_VERBOSE_BATCH,
                 memorize: int = DEEP_MEMORIZE_BATCHES,
                 save_condition: int = DEEP_SAVE_CONDITION_END_EPOCH, # DEEP_SAVE_CONDITION_END_TRAINING to save at the end of training, DEEP_SAVE_CONDITION_END_EPOCH to save at the end of the epoch,
                 overwatch_metric:OverWatchMetric = OverWatchMetric(name=TOTAL_LOSS, condition=DEEP_COMPARE_SMALLER),
                 ):
        self.log_dir = log_dir
        self.verbose = verbose
        self.metrics = metrics
        self.losses = losses
        self.memorize = memorize
        self.save_condition = save_condition
        self.overwatch_metric = overwatch_metric

        # Running metrics
        self.running_total_loss = 0
        self.running_losses = {}
        self.running_metrics = {}

        self.metrics = metrics
        self.train_batches_history = pd.DataFrame(columns=[WALL_TIME, RELATIVE_TIME, EPOCH, BATCH, TOTAL_LOSS] + list(losses.keys()) + list(metrics.keys()))
        self.train_epochs_history = pd.DataFrame(columns=[WALL_TIME, RELATIVE_TIME, EPOCH, TOTAL_LOSS] + list(losses.keys()) + list(metrics.keys()))
        self.validation_history = pd.DataFrame(columns=[WALL_TIME, RELATIVE_TIME, EPOCH, TOTAL_LOSS] + list(losses.keys()) + list(metrics.keys()))

        self.train_batches_history_temp_list = Queue()
        self.train_epochs_history_temp_list = Queue()
        self.validation_history_temp_list = Queue()

        #
        # TEST NEW HISTORY SYSTEM
        #
        train_batches_headers = ",".join([WALL_TIME, RELATIVE_TIME, EPOCH, BATCH, TOTAL_LOSS] + list(losses.keys()) + list(metrics.keys())) + "\n"
        train_epochs_headers = ",".join([WALL_TIME, RELATIVE_TIME, EPOCH,  TOTAL_LOSS] + list(losses.keys()) + list(metrics.keys())) + "\n"
        validation_headers = ",".join([WALL_TIME, RELATIVE_TIME, EPOCH,  TOTAL_LOSS] + list(losses.keys()) + list(metrics.keys())) + "\n"

        print("test")
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

        if self.verbose >= DEEP_VERBOSE_BATCH:
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

        :param minibatch_index->int: Index of the current minibatch
        :param num_minibatches->int: Number of minibatches per epoch
        :param total_loss->int: The total loss
        :param result_losses->dict: List of resulting losses
        :param result_metrics->dict: List of resulting metrics

        RETURN:
        -------

        :return: None
        """
        # Save the running metrics
        self.running_total_loss = self.running_total_loss + total_loss
        self.running_losses = merge_sum_dict(self.running_losses, result_losses)
        self.running_metrics = merge_sum_dict(self.running_metrics, result_metrics)

        # If the user wants to print stats for each batch
        if self.verbose >= DEEP_VERBOSE_BATCH:

            print_metrics = ", ".join(["%s : %f" % (TOTAL_LOSS, total_loss)]
                                      + ["%s : %f" % (loss_name, value.item())
                                         for (loss_name, value) in result_losses.items()]
                                      + ["%s :%f " % (metric_name, value)
                                         for (metric_name, value) in result_metrics.items()])
            Notification(DEEP_NOTIF_RESULT, "[%i/%i] : %s" % (minibatch_index, num_minibatches, print_metrics)).get()

        # Save the data in memory
        if self.memorize == DEEP_MEMORIZE_BATCHES:
            # Save the history in memory
            data = dict([(WALL_TIME, datetime.datetime.now().strftime(TIME_FORMAT)),
                         (RELATIVE_TIME, self.__time()),
                         (EPOCH, epoch_index),
                         (BATCH, minibatch_index)] +
                        [(TOTAL_LOSS, total_loss)] +
                        [(loss_name, value.item()) for (loss_name, value) in result_losses.items()] +
                        [(metric_name, value) for (metric_name, value) in result_metrics.items()])

            self.train_batches_history = self.train_batches_history.append(data, ignore_index=True)
            self.train_batches_history_temp_list.put(data)



        # Save the history
        # Not available for a batch
        # Please do not uncomment
        # self.save()

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

        :param epoch_index->int: current epoch index
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
        if self.verbose >= DEEP_VERBOSE_BATCH:

            print_metrics = ", ".join(["%s : %f" % (TOTAL_LOSS, self.running_total_loss / num_minibatches)]
                                      + ["%s : %f" % (loss_name, value.item() / num_minibatches)
                                         for (loss_name, value) in self.running_losses.items()]
                                      + ["%s : %f" % (metric_name, value / num_minibatches)
                                         for (metric_name, value) in self.running_metrics.items()])
            Notification(DEEP_NOTIF_RESULT, "%s : %s" % (TRAINING, print_metrics))
            
            if self.memorize >= DEEP_MEMORIZE_BATCHES:
              data = dict([(WALL_TIME, datetime.datetime.now().strftime(TIME_FORMAT)),
                           (RELATIVE_TIME, self.__time()),
                          (EPOCH, epoch_index)] +
                          [(TOTAL_LOSS, self.running_total_loss / num_minibatches)] +
                          [(loss_name, value.item() / num_minibatches) for (loss_name, value) in
                           self.running_losses.items()] +
                          [(metric_name, value / num_minibatches) for (metric_name, value) in
                           self.running_metrics.items()])
            self.train_epochs_history = self.train_epochs_history.append(data, ignore_index=True)
            self.train_epochs_history_temp_list.put(data)


        self.running_total_loss = 0
        self.running_losses = {}
        self.running_metrics = {}


        # MANAGE VALIDATION HISTORY
        if total_validation_loss is not None:
            if self.verbose >= DEEP_VERBOSE_BATCH:

                print_metrics = ", ".join(["%s : %f" % (TOTAL_LOSS, total_validation_loss)]
                                          + ["%s : %f" % (loss_name, value.item() / num_minibatches_validation)
                                             for (loss_name, value) in result_validation_losses.items()]
                                          + ["%s : %f" % (metric_name, value / num_minibatches_validation)
                                             for (metric_name, value) in result_validation_metrics.items()])
                Notification(DEEP_NOTIF_RESULT, "%s: %s" % (VALIDATION, print_metrics))

            if self.memorize >= DEEP_MEMORIZE_BATCHES:
                data = dict([(WALL_TIME, datetime.datetime.now().strftime(TIME_FORMAT)),
                             (RELATIVE_TIME, self.__time()),
                             (EPOCH, epoch_index)] +
                             [(TOTAL_LOSS, total_validation_loss / num_minibatches_validation)] +
                            [(loss_name, value.item() / num_minibatches_validation) for (loss_name, value) in result_validation_losses.items()] +
                            [(metric_name, value / num_minibatches_validation) for (metric_name, value) in result_validation_metrics.items()])

                self.validation_history = self.validation_history.append(data, ignore_index=True)
                self.validation_history_temp_list.put(data)

        self.__compute_overwatch_metric(num_minibatches_training = num_minibatches,
                                        running_total_loss=self.running_total_loss,
                                        running_losses=self.running_losses,
                                        running_metrics=self.running_metrics,
                                        total_validation_loss=total_validation_loss,
                                        result_validation_losses=result_validation_losses,
                                        result_validation_metrics=result_validation_metrics)
        Notification(DEEP_NOTIF_SUCCESS, EPOCH_END % (epoch_index, num_epochs))

        self.save()

        #
        # TO BE REMOVE IN THE NEXT VERSION
        #
        if self.__do_saving():
            self.__save_history()

    def on_training_end(self):
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
        #
        # TO BE REMOVED IN THE NEXT VERSION
        #
        self.__save_history()


        Notification(DEEP_NOTIF_SUCCESS, HISTORY_SAVED % self.log_dir)

    def save(self):
        train_batch_history = ""
        train_epochs_history = ""
        validation_history = ""

        for i in range(self.train_batches_history_temp_list.qsize()):
            train_batch_history =  train_batch_history + ",".join(str(value) for (item, value) in self.train_batches_history_temp_list.get().items()) + "\n"

        for i in range(self.train_epochs_history_temp_list.qsize()):
            train_epochs_history =  train_epochs_history + ",".join(str(value) for (item, value) in self.train_epochs_history_temp_list.get().items()) + "\n"

        for i in range(self.validation_history_temp_list.qsize()):
            validation_history =  validation_history + ",".join(str(value) for (item, value) in self.validation_history_temp_list.get().items()) + "\n"

        self.__add_logs("history_train_batches", self.log_dir, DEEP_EXT_CSV, train_batch_history)
        self.__add_logs("history_train_epochs", self.log_dir, DEEP_EXT_CSV, train_epochs_history)
        self.__add_logs("history_validation", self.log_dir, DEEP_EXT_CSV, validation_history)

    #
    # TO BE REMOVED IN THE NEXT VERSION
    #
    def __do_saving(self):
        pass

    def __save_history(self):
        """
        Authors: Alix Leroy
        Save the history into a CSV file
        :return: None
        """
        os.makedirs(self.log_dir, exist_ok=True)
        # Save train batches history
        if self.memorize >= DEEP_MEMORIZE_BATCHES:
            self.train_batches_history.to_csv(self.__get_path(self.train_batches_filename),
                                              header=True, index=True, encoding='utf-8')
        # Save train epochs history
        self.train_epochs_history.to_csv(self.__get_path(self.train_epochs_filename),
                                         header=True, index=True, encoding='utf-8')
        # Save validation history
        self.validation_history.to_csv(self.__get_path(self.validation_filename),
                                       header=True, index=True, encoding='utf-8')

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
        # TODO: Load test history

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


    def __add_logs(self, log_type: str, log_folder: str, log_extension: str, message: str)->None:
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
        l = Logs(log_type, log_folder, log_extension)
        l.add(message)

    def __compute_overwatch_metric(self, num_minibatches_training,
                                        running_total_loss,
                                        running_losses,
                                        running_metrics,
                                        total_validation_loss,
                                        result_validation_losses,
                                        result_validation_metrics)->None:

        # If the validaiton loss is None (No validation) we take the metric from the training as overwatch metric
        if total_validation_loss is None:
            data = dict([(TOTAL_LOSS, running_total_loss / num_minibatches_training)] +
                        [(loss_name, value.item() / num_minibatches_training) for (loss_name, value) in running_losses.items()] +
                        [(metric_name, value / num_minibatches_training) for (metric_name, value) in  running_metrics.items()])

            for key, value in data.items():
                if key == self.overwatch_metric.get_name():
                    self.overwatch_metric.set_value(value)
                    break
        else:
            data = dict([(TOTAL_LOSS, total_validation_loss)] +
                        [(loss_name, value.item()) for (loss_name, value) in result_validation_losses.items()] +
                        [(metric_name, value / num_minibatches_training) for (metric_name, value) in result_validation_metrics.items()])

            for key, value in data.items():
                if key == self.overwatch_metric.get_name():
                    self.overwatch_metric.set_value(value)
                    break

    def get_overwatch_metric(self):
        return copy.deepcopy(self.overwatch_metric)