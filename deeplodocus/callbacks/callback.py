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