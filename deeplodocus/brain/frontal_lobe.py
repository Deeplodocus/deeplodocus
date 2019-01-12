#!/usr/bin/env python3

# Python imports
import inspect

# Back-end imports
from torch import *
import torch
import torch.nn as nn
import torch.nn.functional

# Deeplodocus import
from deeplodocus.core.inference.tester import Tester
from deeplodocus.core.inference.trainer import Trainer
from deeplodocus.core.metrics.loss import Loss
from deeplodocus.core.metrics.metric import Metric
from deeplodocus.core.metrics.over_watch_metric import OverWatchMetric
from deeplodocus.core.model.model import load_model
from deeplodocus.core.optimizer.optimizer import load_optimizer
from deeplodocus.data.dataset import Dataset
from deeplodocus.data.transform_manager import TransformManager
from deeplodocus.utils.flags.msg import *
from deeplodocus.utils.flags.notif import *
from deeplodocus.utils.flags.path import *
from deeplodocus.utils.flags.dtype import *
from deeplodocus.utils.flags.module import *
from deeplodocus.utils.flags.config import DEEP_CONFIG_AUTO
from deeplodocus.utils.generic_utils import get_module
from deeplodocus.utils.generic_utils import get_int_or_float
from deeplodocus.utils.notification import Notification
from deeplodocus.brain.memory.hippocampus import Hippocampus
from deeplodocus.core.metrics import Metrics, Losses


class FrontalLobe(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    The FrontalLabe class works as a model manager.
    This class loads :
        - The model
        - The optimizer
        - The trainer
        - The validator
        - The tester

    This class also allows to :
        - Start the training
        - Evaluate the model on the test dataset
        - Display the summaries

    """
    def __init__(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize the Frontal Lobe

        PARAMETERS:
        -----------

        :param config->Namespace: The config

        RETURN:
        -------

        :return: None
        """
        self.config = None
        self.model = None
        self.trainer = None
        self.validator = None
        self.tester = None
        self.metrics = None
        self.losses = None
        self.optimizer = None
        self.hippocampus = None
        self.device = None

    def set_device(self):
        try:
            if self.config.project.device == DEEP_CONFIG_AUTO:
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(self.config.project.device)
            Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_PROJECT_DEVICE % str(self.device))
        except TypeError:
            Notification(DEEP_NOTIF_FATAL, DEEP_MSG_PROJECT_DEVICE_NOT_FOUND % self.config.project.device)

    def train(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Start the Trainer

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        self.trainer.fit() if self.trainer is not None else Notification(DEEP_NOTIF_ERROR, DEEP_MSG_NO_TRAINER)

    def evaluate(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Start the Tester

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        self.tester.fit() if self.tester is not None else Notification(DEEP_NOTIF_ERROR, DEEP_MSG_NO_TESTER)

    def load(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Load the config into the Frontal Lobe

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        self.load_model()        # Always load model first
        self.load_optimizer()    # Always load the optimizer after the model
        self.load_losses()
        self.load_metrics()
        self.load_trainer()
        self.load_validator()       # Always load the validator before the trainer
        self.load_tester()
        self.load_memory()
        # self.summary()

    def load_model(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Load the model into the Frontal Lobe

        PARAMETERS:
        -----------

        RETURN:
        -------

        :return model->torch.nn.Module:  The model
        """
        # If a module is specified, edit the model name to include the module (for notification purposes)
        model_name = self.config.model.name if self.config.model.module is None \
            else "%s from %s" % (self.config.model.name, self.config.model.module)

        # Notify the user which model is being collected and from where
        Notification(DEEP_NOTIF_INFO, DEEP_MSG_MODEL_LOADING % model_name)

        # Load the model with model.kwargs from the config
        self.model = load_model(**self.config.model.get(),
                                batch_size=self.config.data.dataloader.batch_size)

        # Put model on the required hardware
        self.model.to(self.device)

        # Store the device the model is on for
        self.model.device = self.device

        # Notify the user of success
        Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_MODEL_LOADED % (self.config.model.name, self.model.__module__))

    def load_optimizer(self):
        """
        AUTHORS:
        --------

        :author: Samuel Westlake
        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Load the optimizer with the adequate parameters

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        # Notify the user of which optimizer is being loaded and from where
        optimizer_name = self.config.optimizer.name if self.config.optimizer.module is None \
            else "%s from %s" % (self.config.model.name, self.config.model.module)
        Notification(DEEP_NOTIF_INFO, DEEP_MSG_OPTIM_LOADING % optimizer_name)
        # An optimizer cannot be loaded without a model (self.model.parameters() is required)
        if self.model is not None:
            # Load the optimizer
            self.optimizer = load_optimizer(model_parameters=self.model.parameters(),
                                            **self.config.optimizer.get())
            # Notify the user of success
            Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_OPTIM_LOADED
                         % (self.config.optimizer.name, self.optimizer.__module__))
        else:
            # Notify the user that a model must be loaded
            Notification(DEEP_NOTIF_FATAL, DEEP_MSG_OPTIM_LOADED_FAIL % DEEP_MSG_MODEL_NOT_LOADED)

    def load_losses(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Load the losses

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return loss_functions->dict: The losses
        """
        losses = {}
        if self.config.losses.get():
            for key, config in self.config.losses.get().items():
                loss_name = "%s : %s" % (key, config.name) if config.module is None \
                    else "%s : %s from %s" % (key, config.name, config.module)
                Notification(DEEP_NOTIF_INFO, DEEP_MSG_LOSS_LOADING % loss_name)
                loss = get_module(name=config.name,
                                  module=config.module,
                                  browse=DEEP_MODULE_LOSSES)
                method = loss(**config.kwargs.get())
                # Check the weight
                if self.config.losses.check("weight", key):
                    if get_int_or_float(config.weight) not in (DEEP_TYPE_INTEGER, DEEP_TYPE_FLOAT):
                        Notification(DEEP_NOTIF_FATAL, "The loss function %s doesn't have a correct weight argument" % key)
                else:
                    Notification(DEEP_NOTIF_FATAL, "The loss function %s doesn't have any weight argument" % key)
                # Create the loss
                if isinstance(method, torch.nn.Module):
                    losses[str(key)] = Loss(name=str(key),
                                            weight=float(config.weight),
                                            loss=method)
                    Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_LOSS_LOADED % (key, config.name, loss.__module__))
                else:
                    Notification(DEEP_NOTIF_FATAL, "The loss function %s is not a torch.nn.Module instance" % key)
        else:
            Notification(DEEP_NOTIF_INFO, DEEP_MSG_LOSS_NONE)
        self.losses = Losses(losses)

    def load_metrics(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Load the metrics

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return loss_functions->dict: The metrics
        """
        metrics = {}
        if self.config.metrics.get():
            Notification(DEEP_NOTIF_INFO, DEEP_MSG_METRIC_LOADING % ", ".join(list(self.config.metrics.get().keys())))
            for key, config in self.config.metrics.get().items():
                metric = get_module(name=config.name,
                                    module=config.module,
                                    browse=DEEP_MODULE_METRICS)
                if inspect.isclass(metric):
                    method = metric(**config.kwargs.get())
                else:
                    method = metric
                metrics[str(key)] = Metric(name=str(key), method=method)
                Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_METRIC_LOADED % (key, config.name, metric.__module__))
        else:
            Notification(DEEP_NOTIF_INFO, DEEP_MSG_METRIC_NONE)
        self.metrics = Metrics(metrics)

    def load_trainer(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Load a trainer

        PARAMETERS:
        -----------
        None

        RETURN:
        -------

        :return None
        """
        # If the train step is enabled
        if self.config.data.enabled.train:
            Notification(DEEP_NOTIF_INFO, DEEP_NOTIF_DATA_LOADING % self.config.data.dataset.train.name)

            # Transform Manager
            transform_manager = TransformManager(**self.config.transform.train.get())

            # Dataset
            dataset = Dataset(**self.config.data.dataset.train.get(),
                              transform_manager=transform_manager,
                              cv_library=self.config.project.cv_library)
            # Trainer
            self.trainer = Trainer(**self.config.data.dataloader.get(),
                                   model=self.model,
                                   dataset=dataset,
                                   metrics=self.metrics,
                                   losses=self.losses,
                                   optimizer=self.optimizer,
                                   num_epochs=self.config.training.num_epochs,
                                   initial_epoch=self.config.training.initial_epoch,
                                   shuffle=self.config.training.shuffle,
                                   verbose=self.config.history.verbose,
                                   tester=self.validator)
        else:
            Notification(DEEP_NOTIF_INFO, DEEP_MSG_DATA_DISABLED % self.config.data.dataset.train.name)

    def load_validator(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Load the validation inferer in memory

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        # If the validation step is enabled
        if self.config.data.enabled.validation:
            # Transform Manager
            transform_manager = TransformManager(**self.config.transform.validation.get())

            # Dataset
            dataset = Dataset(**self.config.data.dataset.validation.get(),
                              transform_manager=transform_manager,
                              cv_library=self.config.project.cv_library)

            # Validator
            self.validator = Tester(**self.config.data.dataloader.get(),
                                    model=self.model,
                                    dataset=dataset,
                                    metrics=self.metrics,
                                    losses=self.losses)
        else:
            Notification(DEEP_NOTIF_INFO, DEEP_MSG_DATA_DISABLED % self.config.data.dataset.validation.name)

    def load_tester(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Load the test inferer in memory

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        # If the test step is enabled
        if self.config.data.enabled.test:

            # Transform Manager
            transform_manager = TransformManager(**self.config.transform.test.get())

            # Dataset
            dataset = Dataset(**self.config.data.dataset.test.get(),
                              transform_manager=transform_manager,
                              cv_library=self.config.project.cv_library)
            # Tester
            self.tester = Tester(**self.config.data.dataloader.get(),
                                 model=self.model,
                                 dataset=dataset,
                                 metrics=self.metrics,
                                 losses=self.losses)
        else:
            Notification(DEEP_NOTIF_INFO, DEEP_MSG_DATA_DISABLED % self.config.data.dataset.test.name)

    def load_memory(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Load the memory

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        if self.losses is not None and self.metrics is not None:

            overwatch_metric = OverWatchMetric(**self.config.training.overwatch.get())

            # The hippocampus (brain/memory/hippocampus) temporary  handles the saver and the history
            self.hippocampus = Hippocampus(losses=self.losses,
                                           metrics=self.metrics,
                                           model_name=self.config.model.name,
                                           verbose=self.config.history.verbose,
                                           memorize=self.config.history.memorize,
                                           history_directory=DEEP_PATH_HISTORY,
                                           overwatch_metric=overwatch_metric,
                                           save_model_condition=self.config.training.save_condition,
                                           save_model_directory=DEEP_PATH_SAVE_MODEL,
                                           save_model_method=self.config.training.save_method)

    def summary(self):
        if self.model is not None:
            self.model.summary()
        else:
            Notification(DEEP_NOTIF_INFO, DEEP_MSG_MODEL_NOT_LOADED)
        if self.optimizer is not None:
            self.optimizer.summary()
        else:
            Notification(DEEP_NOTIF_INFO, DEEP_MSG_OPTIM_NOT_LOADED)
        if self.losses is not None:
            self.losses.summary()
        else:
            Notification(DEEP_NOTIF_INFO, DEEP_MSG_LOSS_NOT_LOADED)
        if self.metrics is not None:
            self.metrics.summary()
        else:
            Notification(DEEP_NOTIF_INFO, DEEP_MSG_METRIC_NOT_LOADED)


    @staticmethod
    def __model_has_multiple_inputs(list_inputs):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check if the model has multiple inputs

        PARAMETERS:
        -----------

        :param list_inputs(list): The list of inputs in the network

        RETURN:
        -------

        :return->bool: Whether the model has multiple inputs or not
        """
        if len(list_inputs) >= 2:
            return True
        else:
            return False
