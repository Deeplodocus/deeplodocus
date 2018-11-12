import pandas as pd
import time
import os
import datetime
from typing import Union
import __main__
from deeplodocus.utils.flags import *
from deeplodocus.utils.notification import Notification
from deeplodocus.utils.dict_utils import merge_sum_dict

Num = Union[int, float]


class History(object):
    """
    Authors : Alix Leroy,
    The class stores and manages the history
    """

    def __init__(self,
                 metrics: dict,
                 losses: dict,
                 log_dir: str = "%s/results/history" % os.path.dirname(os.path.abspath(__main__.__file__)),
                 train_batches_filename: str = "history_batches_training.csv",
                 train_epochs_filename: str = "history_epochs_training.csv",
                 validation_filename: str = "history_validation.csv",
                 verbose: int = DEEP_VERBOSE_BATCH,
                 data_to_memorize: int = DEEP_MEMORIZE_BATCHES,
                 save_condition: int = DEEP_SAVE_CONDITION_END_EPOCH, # DEEP_SAVE_CONDITION_END_TRAINING to save at the end of training, DEEP_SAVE_CONDITION_END_EPOCH to save at the end of the epoch,
                 write_logs: bool = True
                 ):
        self.write_logs = write_logs
        self.verbose = verbose
        self.metrics = metrics
        self.losses = losses
        self.data_to_memorize = data_to_memorize
        self.save_condition = save_condition

        # Running metrics
        self.running_total_loss = 0
        self.running_losses = {}
        self.running_metrics = {}

        self.metrics = metrics
        self.train_batches_history = pd.DataFrame(columns=[WALL_TIME, RELATIVE_TIME, EPOCH, TOTAL_LOSS] + list(losses.keys()) + list(metrics.keys()))
        self.train_epochs_history = pd.DataFrame(columns=[WALL_TIME, RELATIVE_TIME, EPOCH, TOTAL_LOSS] + list(losses.keys()) + list(metrics.keys()))
        self.validation_history = pd.DataFrame(columns=[WALL_TIME, RELATIVE_TIME, EPOCH, TOTAL_LOSS] + list(losses.keys()) + list(metrics.keys()))

        self.start_time = 0

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
            Notification(DEEP_NOTIF_INFO, EPOCH_START % (epoch_index, num_epochs), write_logs=self.write_logs).get()

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
            print_metrics = ", ".join(["total loss : " + str(total_loss)] +
                                      [str(loss_name) + " : " + str(value.item()) for (loss_name, value) in result_losses.items()] +
                                      [str(metric_name) + " : " + str(value) for (metric_name, value) in result_metrics.items()])
            # print("[" + str(minibatch_index) + "/" + str(num_minibatches) + "] :  " + str(print_metrics))
            Notification(DEEP_NOTIF_INFO, "[%i/%i] : %s" % (minibatch_index, num_minibatches, print_metrics),
                         write_logs=self.write_logs).get()

        # Save the data in memory
        if self.data_to_memorize == DEEP_MEMORIZE_BATCHES:
            # Save the history in memory
            data = dict([("total loss", total_loss)] +
                        [(loss_name, value.item()) for (loss_name, value) in result_losses.items()] +
                        [(metric_name, value) for (metric_name, value) in result_metrics.items()])
            data[WALL_TIME] = datetime.datetime.now().strftime(TIME_FORMAT)
            data[RELATIVE_TIME] = self.__time()
            data[EPOCH] = epoch_index
            data[BATCH] = minibatch_index
            self.train_batches_history = self.train_batches_history.append(data, ignore_index=True)

        # Save the history
        # Not available for a batch
        # Please do not uncomment
        #if self.__do_saving() is True:
        #    self.__save_history()

    def on_epoch_end(self,
                     epoch_index: int,
                     num_epochs: int,
                     num_minibatches: int,
                     total_validation_loss: int,
                     result_validation_losses: dict,
                     result_validation_metrics: dict,
                     num_minibatches_validation: int):
        """
        Author: Alix Leroy, SW
        Method for managing history at the end of each epoch
        :param epoch_index: int: current epoch index
        :param num_epochs: int: total number of epoch
        :param num_minibatches: int: number of minibatches per epoch
        :param total_validation_loss:
        :param result_validation_losses:
        :param result_validation_metrics:
        :param num_minibatches_validation:
        :return:
        """
        # MANAGE TRAINING HISTORY
        if self.verbose >= DEEP_VERBOSE_BATCH:
            print_metrics = ", ".join(["%s : %f" % (TOTAL_LOSS, self.running_total_loss / num_minibatches)] +
                                      [str(loss_name) + " : " + str(value.item() / num_minibatches) for (loss_name, value) in self.running_losses.items()] +
                                      [str(metric_name) + " : " + str(value / num_minibatches) for (metric_name, value) in self.running_metrics.items()])
            Notification(DEEP_NOTIF_INFO, "%s : %s" % (TRAINING, print_metrics), write_logs=self.write_logs).get()
        if self.data_to_memorize >= DEEP_MEMORIZE_BATCHES:
            data = dict([(TOTAL_LOSS, self.running_total_loss/num_minibatches)] +
                        [(loss_name, value.item()/num_minibatches) for (loss_name, value) in self.running_losses.items()] +
                        [(metric_name, value/num_minibatches) for (metric_name, value) in self.running_metrics.items()])
            data[WALL_TIME] = datetime.datetime.now().strftime(TIME_FORMAT)
            data[RELATIVE_TIME] = self.__time()
            data[EPOCH] = epoch_index
            self.train_epochs_history = self.train_epochs_history.append(data, ignore_index=True)
        self.running_total_loss = 0
        self.running_losses = {}
        self.running_metrics = {}
        # MANAGE VALIDATION HISTORY
        if total_validation_loss is not None:
            if self.verbose >= DEEP_VERBOSE_BATCH:
                print_metrics = ", ".join(["%s : %f" % (TOTAL_LOSS, total_validation_loss)] +
                                          [str(loss_name) + " : " + str(value.item() / num_minibatches_validation) for
                                           (loss_name, value) in result_validation_losses.items()] +
                                          [str(metric_name) + " : " + str(value / num_minibatches_validation) for
                                           (metric_name, value) in result_validation_metrics.items()])
                Notification(DEEP_NOTIF_INFO, "%s: %s" % (VALIDATION, print_metrics), write_logs=self.write_logs).get()
            if self.data_to_memorize >= DEEP_MEMORIZE_BATCHES:
                data = dict([(TOTAL_LOSS, total_validation_loss / num_minibatches_validation)] +
                            [(loss_name, value.item() / num_minibatches_validation) for (loss_name, value) in
                             result_validation_losses.items()] +
                            [(metric_name, value / num_minibatches_validation) for (metric_name, value) in
                             result_validation_metrics.items()])
                data[WALL_TIME] = datetime.datetime.now().strftime(TIME_FORMAT)
                data[RELATIVE_TIME] = self.__time()
                data[EPOCH] = epoch_index
                self.validation_history = self.validation_history.append(data, ignore_index=True)
            Notification(DEEP_NOTIF_SUCCESS, EPOCH_END % (epoch_index, num_epochs), write_logs=self.write_logs).get()
        if self.__do_saving():
            self.__save_history()


    def on_training_end(self):
        """
        Authors: Alix Leroy
        Actions to perform when the training finishes
        :return: None
        """
        self.__save_history()
        Notification(DEEP_NOTIF_SUCCESS, HISTORY_SAVED % self.log_dir, write_logs=self.write_logs).get()


    def __do_saving(self):
        pass
    # TODO : Check if history has to be saved

    def __save_history(self):
        """
        Authors: Alix Leroy
        Save the history into a CSV file
        :return: None
        """
        os.makedirs(self.log_dir, exist_ok=True)

        # Save train batches historyn
        if self.data_to_memorize >= DEEP_MEMORIZE_BATCHES:
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
            self.start_time = time.time() - self.train_epochs_history["relative time"][self.train_epochs_history.index[-1]]

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
        pass
