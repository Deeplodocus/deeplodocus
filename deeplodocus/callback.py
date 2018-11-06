from torch import tensor
from typing import List
from typing import Union

from .callbacks.saver import Saver
from .callbacks.history import History
from .callbacks.stopping import Stopping
from deeplodocus.core.metric import Metric
from deeplodocus.core.loss import Loss


Num = Union[int, float]


class Callback(object):
    """

    save_metric : metric to check for the saving, only useful on auto mode. Loss value by default
    """

    def __init__(self,
                 # History
                 losses: dict,
                 metrics: dict,
                 working_directory,
                 model_name,
                 verbose,
                 data_to_memorize,
                 save_condition,
                 # Stopping
                 stopping_parameters,
                 # Save
                 model,
                 optimizer,
                 write_logs: bool = True
                ):

        self.write_logs=write_logs
        self.model = None

        #
        # HISTORY
        #

        self.metrics = metrics
        self.losses = losses
        self.working_directory = working_directory
        self.model_name = model_name
        self.verbose = verbose
        self.save_condition = save_condition


        #
        # GENERATING THE CALLBACKS
        #

        # Save
        self.__initialize_saver(model=model, optimizer=optimizer)                                 # Callback to save the config, the model and the weights


        # History
        self.__initialize_history(data_to_memorize=data_to_memorize)            # Callback to keep track of the history, display, plot and save it

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
        self.history.on_batch_end(minibatch_index=minibatch_index,
                                  num_minibatches=num_minibatches,
                                  epoch_index=epoch_index,
                                  total_loss=total_loss,
                                  result_losses= result_losses,
                                  result_metrics=result_metrics)



    def on_epoch_end(self, epoch_index:int, num_epochs:int):
        """
        Authors : Alix Leroy,
        Call callbacks at the end of one epoch
        :return: None
        """

        self.history.on_epoch_end(epoch_index=epoch_index, num_epochs=num_epochs)
        self.saver.on_epoch_end()
        self.stopping.on_epoch_end()


    def on_training_end(self):
        """
        Authors : Alix Leroy,
        Calls callbacks at the end of the training
        :return: None
        """

        self.history.on_training_end()
        self.saver.on_training_end()
        self.stopping.on_training_end()


    def __initialize_history(self, data_to_memorize:int)->None:
        """
        Authors : Samuel Westlake, Alix Leroy
        Initialise the history
        :return: None
        """

        # Get the directory for saving the history
        log_dir = "%s/%s" % (self.working_directory, self.model_name)

        # Initialize the history
        self.history = History(metrics=self.metrics,
                               losses=self.losses,
                               log_dir=log_dir,
                               train_batches_filename="%s_history_train_batches.csv" % self.model_name,
                               train_epochs_filename="%s_history_train_epochs.csv" % self.model_name,
                               validation_filename="%s_history_validation.csv" % self.model_name,
                               verbose=self.verbose,
                               data_to_memorize=data_to_memorize,
                               save_condition = self.save_condition,
                               write_logs=self.write_logs)

    def update(self):

        self.history.update(num_epochs=self.num_epochs, num_batches=self.num_batches)


    def __initialize_saver(self, model, optimizer):
        """
        Authors : Alix Leroy,
        Initialize the saver
        :param model: model to save
        :return: None
        """
        self.saver = Saver(model, optimizer)

    def pause(self):

        print("Callbacks pause not implemented")