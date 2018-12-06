from typing import Union

import datetime

from torch.nn import Module

from deeplodocus.callbacks.saver import Saver
from deeplodocus.callbacks.history import History
from deeplodocus.callbacks.stopping import Stopping
from deeplodocus.utils.flags import *
from deeplodocus.core.metrics.over_watch_metric import OverWatchMetric

Num = Union[int, float]


class Callback(object):
    """

    save_metric : metric to check for the saving, only useful on auto mode. Loss value by default
    """

    def __init__(self,
                 # History
                 losses: dict,
                 metrics: dict,

                 model_name:str,
                 verbose:int,
                 memorize:int,
                 # Saver
                 save_condition:int = DEEP_SAVE_CONDITION_AUTO,
                 save_model_method:int = DEEP_SAVE_NET_FORMAT_PYTORCH,
                 history_directory: str = DEEP_PATH_HISTORY,
                 save_directory: str = DEEP_PATH_SAVE_MODEL,
                 overwatch_metric: OverWatchMetric = OverWatchMetric(name=TOTAL_LOSS, condition=DEEP_COMPARE_SMALLER),
                 # Stopping
                 stopping_parameters=None,
                ):

        self.model = None

        #
        # HISTORY
        #

        self.metrics = metrics
        self.losses = losses
        self.history_directory = history_directory
        self.save_directory = save_directory
        self.model_name = model_name
        self.verbose = verbose
        self.save_condition = save_condition
        self.save_model_method = save_model_method
        self.overwatch_metric = overwatch_metric


        #
        # GENERATING THE CALLBACKS
        #

        # Save
        self.__initialize_saver()                                 # Callback to save the config, the model and the weights


        # History
        self.__initialize_history(memorize=memorize)            # Callback to keep track of the history, display, plot and save it

        # Stopping
        self.stopping = Stopping(stopping_parameters)                                     # Callback to check the condition to stop the training



    def on_train_begin(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Calls callback at the beginning of the training

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        self.history.on_train_begin()

    def on_epoch_start(self, epoch_index: int, num_epochs: int):
        """
        Author: SW
        :param epoch_index: int: index of current epoch
        :param num_epochs: int: total number of epochs
        :return:
        """
        self.history.on_epoch_start(epoch_index, num_epochs)

    def on_batch_end(self, minibatch_index:int, num_minibatches:int, epoch_index:int, total_loss:int, result_losses:dict, result_metrics:dict):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Calls callbacks at the end of each minibatch

        PARAMETERS:
        -----------

        :param minibatch_index->int: The index of the current minibatch
        :param num_minibatches->int: The number of batches per epoch
        :param epoch_index->int: Index of the current epoch
        :param total_loss->int: The total loss
        :param result_losses->dict: The dict of resulting losses
        :param result_metrics-dict: The dict of resulting metrics

        RETURN:
        -------

        :return: None
        """
        # self.history.on_batch_end(minibatch_index=minibatch_index,
        #                           num_minibatches=num_minibatches,
        #                           epoch_index=epoch_index,
        #                           total_loss=total_loss,
        #                           result_losses= result_losses,
        #                           result_metrics=result_metrics)



    def on_epoch_end(self, epoch_index:int, num_epochs:int, num_minibatches:int, model:Module, total_validation_loss:int, result_validation_losses:dict, result_validation_metrics:dict, num_minibatches_validation:int):
        """
        Authors : Alix Leroy,
        Call callbacks at the end of one epoch
        :return: None
        """

        # self.history.on_epoch_end(epoch_index=epoch_index,
        #                           num_epochs=num_epochs,
        #                           num_minibatches=num_minibatches,
        #                           total_validation_loss=total_validation_loss,
        #                           result_validation_losses=result_validation_losses,
        #                           result_validation_metrics=result_validation_metrics,
        #                           num_minibatches_validation=num_minibatches_validation)
        current_overwatch_metric = self.history.get_overwatch_metric()
        self.saver.on_epoch_end(model, current_overwatch_metric=current_overwatch_metric)
        self.stopping.on_epoch_end()


    def on_training_end(self, model:Module):
        """
        Authors : Alix Leroy,
        Calls callbacks at the end of the training
        :return: None
        """

        #self.history.on_training_end()
        self.saver.on_training_end(model=model)
        self.stopping.on_training_end()


    def __initialize_history(self, memorize:int)->None:
        """
        Authors : Samuel Westlake, Alix Leroy
        Initialise the history
        :return: None
        """

        timestr = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        train_batches_filename =  self.model_name + "_history_train_batches-"+ timestr + ".csv"
        train_epochs_filename =  self.model_name + "_history_train_epochs-"+ timestr + ".csv"
        validation_filename =  self.model_name + "_history_validation-"+ timestr + ".csv"


        # Initialize the history
        self.history = History(metrics=self.metrics,
                               losses=self.losses,
                               log_dir=self.history_directory,
                               train_batches_filename=train_batches_filename,
                               train_epochs_filename=train_epochs_filename,
                               validation_filename=validation_filename,
                               verbose=self.verbose,
                               memorize=memorize,
                               save_condition = self.save_condition)

    def update(self):
        self.history.update(num_epochs=self.num_epochs, num_batches=self.num_batches)


    def __initialize_saver(self):
        """
        Authors : Alix Leroy,
        Initialize the saver
        :param model: model to save
        :return: None
        """
        self.saver = Saver(model_name=self.model_name,
                           save_condition=self.save_condition,
                           save_model_method=self.save_model_method)

    def pause(self):
        print("Callbacks pause not implemented")