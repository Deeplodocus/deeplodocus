from .callbacks.save import Save
from .callbacks.history import History
from .callbacks.stopping import Stopping


class Callback(object):
    """

    save_metric : metric to check for the saving, only useful on auto mode. Loss value by default
    """

    def __init__(self,
                 # History
                 metrics,
                 initial_epoch,
                 working_directory,
                 model_name,
                 verbose,
                 save_condition,
                 # Stopping
                 stopping_parameters,
                 # Save
                 model,
                 input_size,
                 input_names,
                 output_names,
                ):

        self.model = None

        #
        # HISTORY
        #

        self.metrics = metrics
        self.initial_epoch = initial_epoch
        self.working_directory = working_directory
        self.model_name = model_name
        self.verbose = verbose
        self.save_condition = save_condition


        #
        # GENERATING THE CALLBACKS
        #

        # Save
        self.__initialize_saver(model)                                 # Callback to save the config, the model and the weights


        # History
        self.__initialize_history()            # Callback to keep track of the history, display, plot and save it

        # Stopping
        self.stopping = Stopping(**stopping_parameters)                                     # Callback to check the condition to stop the training



    def on_train_begin(self):
        """
        Authors : Alix Leroy,
        Calls callback at the beginning of the training
        :return:
        """
        self.history.on_train_begin()


    def on_batch_end(self):
        """
        Authors : Alix Leroy,
        Calls callbacks at the end of one epoch
        :return: None
        """
        self.history.on_batch_end()
        self.stopping.on_batch_end()


    def on_epoch_end(self):
        """
        Authors : Alix Leroy,
        Call callbacks at the end of one epoch
        :return: None
        """

        self.history.on_epoch_end()
        self.save.on_epoch_end()
        self.stopping.on_epoch_end()


    def on_training_end(self):
        """
        Authors : Alix Leroy,
        Calls callbacks at the end of the training
        :return: None
        """

        self.history.on_training_end()
        self.save.on_training_end()
        self.stopping.on_training_end()


    def __initialize_history(self):
        """
        Authors : Samuel Westlake, Alix Leroy
        Initialise the history
        :return: None
        """

        # Get the directory for saving the history
        log_dir = "%s/%s" % (self.working_dir, self.model_name)

        # Initialize the history
        self.history = History(metrics=self.metrics,
                               initial_epoch=self.initial_epoch,
                               log_dir=log_dir,
                               file_name="%s_history.csv" % self.model_name,
                               verbose=self.verbose,
                               save_condition = self.save_condition)


    def __initialize_saver(self, model):
        """
        Authors : Alix Leroy,
        Initialize the saver
        :param model: model to save
        :return: None
        """
        self.saver = Saver(model)