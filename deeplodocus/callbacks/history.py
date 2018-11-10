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
                 log_dir: str = os.path.dirname(os.path.abspath(__main__.__file__))+ "/results/history/",
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
        self.train_batches_history = pd.DataFrame(columns=["wall time", "relative time", "epoch", "batch", "total loss"] + list(losses.keys()) + list(metrics.keys()))
        self.train_epochs_history = pd.DataFrame(columns=["wall time", "relative time", "epoch", "total loss"] + list(losses.keys()) + list(metrics.keys()))
        self.validation_history = pd.DataFrame(columns=["wall time", "relative time", "epoch", "total loss"] + list(losses.keys()) + list(metrics.keys()))

        self.start_time = 0

        # Filepaths
        self.train_batches_filepath = "%s/%s" % (log_dir, train_batches_filename)
        self.train_epochs_filepath = "%s/%s" % (log_dir, train_epochs_filename)
        self.validation_filepath = "%s/%s" % (log_dir, validation_filename)

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


    def on_batch_end(self, minibatch_index:int, num_minibatches:int, epoch_index:int, total_loss:int, result_losses:dict, result_metrics:dict):
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
            print("[" + str(minibatch_index) + "/" + str(num_minibatches) + "] :  " + str(print_metrics))

        # Save the data in memory
        if self.data_to_memorize == DEEP_MEMORIZE_BATCHES:
            # Save the history in memory
            data = dict([("total loss", total_loss)] +
                        [(loss_name, value.item()) for (loss_name, value) in result_losses.items()] +
                        [(metric_name, value) for (metric_name, value) in result_metrics.items()])
            data["wall time"] = datetime.datetime.now().strftime("%Y:%m:%d:%H:%M:%S")
            data["relative time"] = self.__time()
            data["epoch"] = epoch_index
            data["batch"] = minibatch_index
            self.train_batches_history = self.train_batches_history.append(data, ignore_index=True)

        # Save the history
        # Not available for a batch
        # Please do not uncomment
        #if self.__do_saving() is True:
        #    self.__save_history()

    def on_epoch_end(self, epoch_index:int, num_epochs:int, num_minibatches:int, total_validation_loss:int, result_validation_losses:dict, result_validation_metrics:dict, num_minibatches_validation:int):
        """
        Authors : Alix Leroy,
        Called at the end of every epoch of the training
        :return: None
        """

        #
        # MANAGE TRAIN HISTORY
        #
        # If we want to display the metrics at the end of each epoch
        if self.verbose >= DEEP_VERBOSE_BATCH:
            print_metrics = ", ".join(["total loss : " + str(self.running_total_loss / num_minibatches)] +
                                      [str(loss_name) + " : " + str(value.item() / num_minibatches) for (loss_name, value) in self.running_losses.items()] +
                                      [str(metric_name) + " : " + str(value / num_minibatches) for (metric_name, value) in self.running_metrics.items()])

            print("==============================================================================================================================")
            print("Summary Epoch " + str(epoch_index) + "/" + str(num_epochs) + " : ")
            print("Training : " + str(print_metrics))




        # Save the data in memory
        if self.data_to_memorize >= DEEP_MEMORIZE_BATCHES:
            # Save the history in memory
            data = dict([("total loss", self.running_total_loss/num_minibatches)] +
                        [(loss_name, value.item()/num_minibatches) for (loss_name, value) in self.running_losses.items()] +
                        [(metric_name, value/num_minibatches) for (metric_name, value) in self.running_metrics.items()])
            data["wall time"] = datetime.datetime.now().strftime("%Y:%m:%d:%H:%M:%S")
            data["relative time"] = self.__time()
            data["epoch"] = epoch_index
            self.train_epochs_history = self.train_epochs_history.append(data, ignore_index=True)


        self.running_total_loss = 0
        self.running_losses = {}
        self.running_metrics = {}

        #
        # MANAGE VALIDATION HISTORY
        #

        if total_validation_loss is not None:

            # If we want to display the metrics at the end of each epoch
            if self.verbose >= DEEP_VERBOSE_BATCH:
                print_metrics = ", ".join(["total loss : " + str(total_validation_loss)] +
                                          [str(loss_name) + " : " + str(value.item() / num_minibatches_validation) for
                                           (loss_name, value) in result_validation_losses.items()] +
                                          [str(metric_name) + " : " + str(value / num_minibatches_validation) for
                                           (metric_name, value) in result_validation_metrics.items()])

                print("Validation : " + str(print_metrics))

            # Save the data in memory
            if self.data_to_memorize >= DEEP_MEMORIZE_BATCHES:
                # Save the history in memory
                data = dict([("total loss", total_validation_loss / num_minibatches_validation)] +
                            [(loss_name, value.item() / num_minibatches_validation) for (loss_name, value) in
                             result_validation_losses.items()] +
                            [(metric_name, value / num_minibatches_validation) for (metric_name, value) in
                             result_validation_metrics.items()])
                data["wall time"] = datetime.datetime.now().strftime("%Y:%m:%d:%H:%M:%S")
                data["relative time"] = self.__time()
                data["epoch"] = epoch_index
                self.validation_history = self.validation_history.append(data, ignore_index=True)

        print("==============================================================================================================================")

        #
        # END EPOCH PARAMETERS
        #

        # Save the history
        if self.__do_saving() is True:
            self.__save_history()



    def on_training_end(self):
        """
        Authors: Alix Leroy
        Actions to perform when the training finishes
        :return: None
        """

        self.__save_history()
        Notification(DEEP_NOTIF_SUCCESS, "History saved", write_logs=self.write_logs)



    def __do_saving(self):
        pass
    # TODO : Check if history has to be saved


    def __save_history(self):
        """
        Authors: Alix Leroy
        Save the history into a CSV file
        :return: None
        """

        # Save train batches history
        if self.data_to_memorize >= DEEP_MEMORIZE_BATCHES:
            self.train_batches_history.to_csv(self.train_batches_filepath, header=True, index=True, encoding='utf-8')

        # Save train epochs history
        self.train_epochs_history.to_csv(self.train_epochs_filepath, header=True, index=True, encoding='utf-8')

        # Save validation history
        self.validation_history.to_csv(self.validation_filepath, header=True, index=True, encoding='utf-8')



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
        if os.path.isfile(self.train_batches_filepath):
            self.train_batches_history = pd.read_csv(self.train_batches_filepath)

        # Load train epochs history
        if os.path.isfile(self.train_epochs_filepath):
            self.train_epochs_history = pd.read_csv(self.train_epochs_filepath)

        # Load Validation history
        if os.path.isfile(self.validation_filepath):
            self.validation_history = pd.read_csv(self.validation_filepath)

        # TODO: Load test history

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

