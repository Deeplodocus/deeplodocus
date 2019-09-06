#!/usr/bin/env python3

# Python imports
import inspect

# Back-end imports
import torch.nn.functional

# Deeplodocus import
from deeplodocus.brain.memory.hippocampus import Hippocampus
from deeplodocus.brain.signal import Signal
from deeplodocus.brain.thalamus import Thalamus
from deeplodocus.callbacks.printer import Printer
from deeplodocus.core.inference.tester import Tester
from deeplodocus.core.inference.predictor import Predictor
from deeplodocus.core.inference.trainer import Trainer
from deeplodocus.core.metrics import Metrics, Losses
from deeplodocus.core.metrics.loss import Loss
from deeplodocus.core.metrics.metric import Metric
from deeplodocus.core.metrics.over_watch_metric import OverWatchMetric
from deeplodocus.core.model.model import load_model
from deeplodocus.core.optimizer.optimizer import load_optimizer
from deeplodocus.data.load.dataset import Dataset
from deeplodocus.data.transform.output import OutputTransformer
from deeplodocus.data.transform.transform_manager import TransformManager
from deeplodocus.utils.generic_utils import get_module
from deeplodocus.utils.generic_utils import get_int_or_float
from deeplodocus.utils.generic_utils import get_corresponding_flag
from deeplodocus.utils.notification import Notification

# Deeplodocus flags
from deeplodocus.flags import *


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

    PUBLIC METHODS:
    ---------------

    :method set_device: Set the device to be used for training or inference
    :method train: Start the training of the network
    :method test: Start the test of the network on the Test set
    :method validate: Start the test of the network on the Validation set
    :method continue_training: Continue the training of the network
    :method predict: Predict outputs of the data in the prediction set
    :method load: Load the content of the config into the Frontal Lobe
    :method load_model:
    :method load_optimizer:
    :method load_trainer:
    :method load_tester:
    :method load_validator:
    :method load_predicto:
    :method load_losses:
    :method load_metrics:
    :method load_memory:
    :method summary:

    PRIVATE METHODS:
    ----------------

    :method __init__: Initialize the frontal lobe of Deeplodocus
    :method __load_checkpoint:
    :method __load_model:
    :method __model_has_multiple_inputs: Check wether the model has multiple inputs or not
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

        :param config(Namespace): The config

        RETURN:
        -------

        :return: None
        """
        self.config = None
        self.model = None
        self.trainer = None
        self.validator = None
        self.tester = None
        self.predictor = None
        self.metrics = None
        self.losses = None
        self.optimizer = None
        self.hippocampus = None
        self.device = None
        self.device_ids = None
        self.printer = Printer()

    def set_device(self):
        """
        AUTHORS:
        --------

        Author: Samuel Westlake

        DESCRIPTION:
        ------------

        Sets self.device and self.device_ids, depending on the config specifications and available hardware.

        If multiple device ids are specified (or found when device_ids = "auto"), the output_device, self.device
        will be self.device_ids[0]. Note that nn.DataParallel uses this as the output device by default.

        RETURN:
        -------

        :return: None
        """
        # If device_ids is auto, grab all available devices, else use devices specified
        if self.config.project.device_ids == "auto":
            self.device_ids = list(range(torch.cuda.device_count()))
        else:
            self.device_ids = self.config.project.device_ids

        # If device is auto, set as 'cuda:x' or cpu as appropriate, else use specified value
        try:
            if self.config.project.device == "auto":
                self.device = torch.device("cuda:%i" % self.device_ids[0] if torch.cuda.is_available() else "cpu")
            else:
                if self.config.project.device == "cuda":
                    device = "cuda:%i" % self.device_ids[0]
                else:
                    device = self.config.project.device
                self.device = torch.device(device)
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

    def continue_training(self, epochs=None):
        """
        :return:
        """
        self.trainer.continue_training(epochs=epochs)

    def test(self):
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
        if self.tester is None:
            Notification(DEEP_NOTIF_FATAL, DEEP_MSG_NO_TESTER)
        else:
            total_loss, losses, metrics = self.tester.evaluate(self.model)
            self.printer.validation_epoch_end(losses, total_loss, metrics)

    def validate(self):
        if self.validator is None:
            Notification(DEEP_NOTIF_FATAL, DEEP_MSG_NO_VALIDATOR)
        else:
            total_loss, losses, metrics = self.validator.evaluate(self.model)
            self.printer.validation_epoch_end(losses, total_loss, metrics)

    def predict(self):
        if self.predictor is None:
            Notification(DEEP_NOTIF_FATAL, DEEP_MSG_NO_PREDICTOR)
        else:
            self.predictor.predict(self.model)

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
        self.load_validator()       # Always load the validator before the trainer
        self.load_tester()
        self.load_trainer()
        self.load_predictor()
        self.load_memory()

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
        model = None

        checkpoint = self.__load_checkpoint()

        if self.config.model.from_file:
            # If model name, origin and state_dict are all specified in the checkpoint
            if all(key in checkpoint for key in ("name", "origin", "model_state_dict")):
                model = self.__load_model(
                    name=checkpoint["name"],
                    module=checkpoint["origin"],
                    device=self.device,
                    device_ids=self.device_ids,
                    batch_size=self.config.data.dataloader.batch_size,
                    **self.config.model.get_all(ignore=["from_file", "file", "name", "module"]),
                    model_state_dict=checkpoint["model_state_dict"],
                    weights_path=self.config.model.file,
                    notif=DEEP_NOTIF_WARNING
                )
            elif model is None:
                name = self.config.model.name
                origin = self.config.model.module
                model_state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
                model = self.__load_model(
                    name=name,
                    module=origin,
                    device=self.device,
                    device_ids=self.device_ids,
                    batch_size=self.config.data.dataloader.batch_size,
                    **self.config.model.get_all(ignore=["from_file", "file", "name", "module"]),
                    model_state_dict=model_state_dict,
                    weights_path=self.config.model.file,
                    notif=DEEP_NOTIF_WARNING
                )
        else:
            model = self.__load_model(
                device=self.device,
                device_ids=self.device_ids,
                batch_size=self.config.data.dataloader.batch_size,
                **self.config.model.get_all(ignore=["from_file", "file"]),
                notif=DEEP_NOTIF_FATAL
            )
        self.model = model

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
        # If a module is specified, edit the optimizer name to include the module (for notification purposes)
        if self.config.optimizer.module is None:
            optimizer_path = "%s from default modules" % self.config.optimizer.name
        else:
            optimizer_path = "%s from %s" % (self.config.optimizer.name, self.config.optimizer.module)

        # Notify the user which model is being collected and from where
        Notification(DEEP_NOTIF_INFO, DEEP_MSG_OPTIM_LOADING % optimizer_path)

        # An optimizer cannot be loaded without a model (self.model.parameters() is required)
        if self.model is not None:
            # Load the optimizer
            optimizer = load_optimizer(
                model_parameters=self.model.parameters(),
                **self.config.optimizer.get()
            )
            msg = "%s from %s" % (self.config.optimizer.name, optimizer.module)
            if self.config.model.from_file:
                checkpoint = self.__load_checkpoint()
                if "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    msg += " with state dict from %s" % self.config.model.file

            # If model exists, load the into the frontal lobe
            if optimizer is None:
                Notification(DEEP_NOTIF_FATAL, DEEP_MSG_OPTIM_NOT_FOUND % optimizer_path)
            else:
                self.optimizer = optimizer
                Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_OPTIM_LOADED % msg)

        # Notify the user that a model must be loaded
        else:
            Notification(DEEP_NOTIF_FATAL, DEEP_MSG_OPTIM_MODEL_NOT_LOADED)

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

                # Get the expected loss path (for notification purposes)
                if config.module is None:
                    loss_path = "%s : %s from default modules" % (key, config.name)
                else:
                    loss_path = "%s : %s from %s" % (key, config.name, config.module)

                # Notify the user which loss is being collected and from where
                Notification(DEEP_NOTIF_INFO, DEEP_MSG_LOSS_LOADING % loss_path)

                # Get the loss object
                loss, module = get_module(
                    name=config.name,
                    module=config.module,
                    browse=DEEP_MODULE_LOSSES
                )
                method = loss(**config.kwargs.get())

                # Check the weight
                if self.config.losses.check("weight", key):
                    if get_corresponding_flag(flag_list=[DEEP_DTYPE_INTEGER, DEEP_DTYPE_FLOAT],
                                              info=get_int_or_float(config.weight), fatal=False) is None:
                        Notification(DEEP_NOTIF_FATAL, "The loss function %s doesn't have a correct weight argument" % key)
                else:
                    Notification(DEEP_NOTIF_FATAL, "The loss function %s doesn't have any weight argument" % key)

                # Create the loss
                if isinstance(method, torch.nn.Module):
                    losses[str(key)] = Loss(
                        name=str(key),
                        weight=float(config.weight),
                        loss=method
                    )
                    Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_LOSS_LOADED % (key, config.name, module))
                else:
                    Notification(DEEP_NOTIF_FATAL, DEEP_MSG_LOSS_NOT_TORCH % (key, config.name, module))
            self.losses = Losses(losses)
        else:
            Notification(DEEP_NOTIF_INFO, DEEP_MSG_LOSS_NONE)

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
            for key, config in self.config.metrics.get().items():

                # Get the expected loss path (for notification purposes)
                if config.module is None:
                    metric_path = "%s : %s from default modules" % (key, config.name)
                else:
                    metric_path = "%s : %s from %s" % (key, config.name, config.module)

                # Notify the user which loss is being collected and from where
                Notification(DEEP_NOTIF_INFO, DEEP_MSG_METRIC_LOADING % metric_path)

                # Get the metric object
                metric, module = get_module(
                    name=config.name,
                    module=config.module,
                    browse={**DEEP_MODULE_METRICS, **DEEP_MODULE_LOSSES},
                    silence=True
                )

                # If metric is not found by get_module
                if metric is None:
                    Notification(DEEP_NOTIF_FATAL, DEEP_MSG_METRIC_NOT_FOUND % config.name)

                # Check if the metric is a class or a stand-alone function
                if inspect.isclass(metric):
                    method = metric(**config.kwargs.get())
                else:
                    method = metric

                # Add to the dictionary of metrics and notify of success
                metrics[str(key)] = Metric(name=str(key), method=method)
                metrics[str(key)] = Metric(name=str(key), method=method)
                Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_METRIC_LOADED % (key, config.name, module))
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

            # Input Transform Manager
            transform_manager = TransformManager(
                **self.config.transform.train.get(ignore="outputs")
            )

            # Output Transformer
            output_transform_manager = OutputTransformer(
                transform_files=self.config.transform.train.get("outputs")
            )
            # output_transformer.summary()

            # Dataset
            dataset = Dataset(**self.config.data.dataset.train.get(),
                              transform_manager=transform_manager)
            # Trainer
            self.trainer = Trainer(
                **self.config.data.dataloader.get(),
                model=self.model,
                dataset=dataset,
                metrics=self.metrics,
                losses=self.losses,
                optimizer=self.optimizer,
                num_epochs=self.config.training.num_epochs,
                initial_epoch=self.config.training.initial_epoch,
                shuffle_method=self.config.training.shuffle,
                verbose=self.config.history.verbose,
                tester=self.validator,
                transform_manager=output_transform_manager
            )
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
            Notification(DEEP_NOTIF_INFO, DEEP_NOTIF_DATA_LOADING % self.config.data.dataset.validation.name)

            # Transform Manager
            transform_manager = TransformManager(
                **self.config.transform.validation.get(ignore="outputs")
            )

            # Output Transformer
            output_transform_manager = OutputTransformer(
                transform_files=self.config.transform.validation.get("outputs")
            )
            # output_transformer.summary()

            # Dataset
            dataset = Dataset(**self.config.data.dataset.validation.get(),
                              transform_manager=transform_manager)

            # Validator
            self.validator = Tester(
                **self.config.data.dataloader.get(),
                model=self.model,
                dataset=dataset,
                metrics=self.metrics,
                losses=self.losses,
                transform_manager=output_transform_manager
            )
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
            Notification(DEEP_NOTIF_INFO, DEEP_NOTIF_DATA_LOADING % self.config.data.dataset.test.name)

            # Input Transform Manager
            transform_manager = TransformManager(
                **self.config.transform.test.get(ignore="outputs")
            )

            # Output Transformer
            output_transform_manager = OutputTransformer(
                transform_files=self.config.transform.test.get("outputs")
            )
            # output_transformer.summary()

            # Dataset
            dataset = Dataset(**self.config.data.dataset.test.get(),
                              transform_manager=transform_manager)
            # Tester
            self.tester = Tester(
                **self.config.data.dataloader.get(),
                model=self.model,
                dataset=dataset,
                metrics=self.metrics,
                losses=self.losses,
                transform_manager=output_transform_manager
            )
        else:
            Notification(DEEP_NOTIF_INFO, DEEP_MSG_DATA_DISABLED % self.config.data.dataset.test.name)

    def load_predictor(self):
        # If the predict step is enabled
        if self.config.data.enabled.predict:
            Notification(DEEP_NOTIF_INFO, DEEP_NOTIF_DATA_LOADING % self.config.data.dataset.predict.name)

            # Input Transform Manager
            transform_manager = TransformManager(
                **self.config.transform.predict.get(ignore="outputs")
            )

            # Output Transform Manager
            output_transform_manager = OutputTransformer(
                transform_files=self.config.transform.predict.get("outputs")
            )

            # Dataset
            dataset = Dataset(
                **self.config.data.dataset.predict.get(),
                transform_manager=transform_manager)
            # Predictor
            self.predictor = Predictor(
                **self.config.data.dataloader.get(),
                model=self.model,
                dataset=dataset,
                transform_manager=output_transform_manager
            )
        else:
            Notification(DEEP_NOTIF_INFO, DEEP_MSG_DATA_DISABLED % self.config.data.dataset.predict.name)

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

            history_directory = "/".join(
                (get_main_path(), self.config.project.session, "history")
            )
            weights_directory = "/".join(
                (get_main_path(), self.config.project.session, "weights")
            )

            self.hippocampus = Hippocampus(
                losses=self.losses,
                metrics=self.metrics,
                model_name=self.config.model.name,
                verbose=self.config.history.verbose,
                memorize=self.config.history.memorize,
                history_directory=history_directory,
                overwatch_metric=overwatch_metric,
                **self.config.training.saver.get(),
                save_model_directory=weights_directory
            )

    def save_model(self):
        Thalamus().add_signal(
            signal=Signal(
                event=DEEP_EVENT_SAVE_MODEL,
                args={}
            )
        )

    def summary(self):
        """
        AUTHORS:
        --------

        :author: samuel Westlake

        DESCRIPTION:
        ------------

        Print a summary of the model, optimizer, losses and metrics.

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        if self.model is not None:
            self.model.summary()
        else:
            Notification(DEEP_NOTIF_INFO, DEEP_MSG_OPTIM_MODEL_NOT_LOADED)
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

    def __load_checkpoint(self):
        # If loading from file, load data from the given path
        try:
            return torch.load(self.config.model.file, map_location=self.device) if self.config.model.from_file else None
        except AttributeError:
            Notification(
                DEEP_NOTIF_FATAL,
                DEEP_MSG_MODEL_NO_FILE,
                solutions=[
                    "Enter a path to a model file in config/model/file",
                    "Disable load model from file by setting config/model/from_file to False",
                ]
            )
        except FileNotFoundError:
            Notification(DEEP_NOTIF_FATAL, DEEP_MSG_MODEL_FILE_NOT_FOUND % self.config.model.file)

    @staticmethod
    def __load_model(
            name, module, device, device_ids,
            batch_size=None,
            model_state_dict=None,
            notif=DEEP_NOTIF_WARNING,
            weights_path="Unknown",
            **kwargs
    ):
        """
        :param name:
        :param module:
        :param device:
        :param device_ids:
        :param batch_size:
        :param model_state_dict:
        :param notif:
        :param kwargs:
        :return:
        """
        if module is None:
            model_path = "%s from default modules" % name
        else:
            model_path = "%s from %s" % (name, module)

        Notification(DEEP_NOTIF_INFO, DEEP_MSG_MODEL_LOADING % model_path)
        model = load_model(
            name=name,
            module=module,
            device=device,
            device_ids=device_ids,
            batch_size=batch_size,
            model_state_dict=model_state_dict,
            **kwargs
        )
        if model is None:
            Notification(notif, DEEP_MSG_MODEL_NOT_FOUND % model_path)
        else:
            model_path = "%s from %s" % (model.name, model.origin)
            if model_state_dict is not None:
                model_path += " with weights from %s" % weights_path
            Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_MODEL_LOADED % model_path)
        return model

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
