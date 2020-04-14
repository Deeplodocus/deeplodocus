# Python imports
import weakref
from math import ceil

# Backend imports
import torch.nn as nn
from torch.utils.data import DataLoader


# Deeplodocus imports
from deeplodocus.brain.signal import Signal
from deeplodocus.brain.thalamus import Thalamus
from deeplodocus.core.metrics import Metrics
from deeplodocus.data.load.dataset import Dataset
from deeplodocus.core.inference.tester import Tester
from deeplodocus.flags import *
from deeplodocus.utils.namespace import Namespace
from deeplodocus.utils.generic_utils import get_corresponding_flag
from deeplodocus.utils.notification import Notification


class Trainer(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Trainer instance to train a model

    PUBLIC METHOD:
    --------------
    :method fit: Start the training
    :method detach: Detach the output tensors from the model in order to avoid memory leaks
    :method continue_training: Continue the training of the model
    :method saving_required: Send a signal to the Saver in order to save the model
    :method send_save_params: Send the parameters to the Saver

    PRIVATE METHOD:
    ---------------
    :method __init__: Initialize the Trainer
    :method __train: Loop over the dataset to train the network
    :method  __evaluate_epoch: Evaluate the model using the Validator

    """

    """
    "
    " PRIVATE METHODS
    "
    """

    def __init__(
            self,
            dataset,
            name="Trainer",
            model=None,
            optimizer=None,
            losses=None,
            metrics=None,
            num_epochs: int = 10,
            initial_epoch: int = 1,
            batch_size: int = 4,
            num_workers: int = 4,
            shuffle_method: Flag = DEEP_SHUFFLE_NONE,
            verbose: Flag = DEEP_VERBOSE_BATCH,
            tester: Tester = None,
            transform_manager=None
    ) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize a Trainer instance

        PARAMETERS:
        -----------

        :param model (torch.nn.Module): The model which has to be trained
        :param dataset (Dataset): The dataset to be trained on
        :param metrics (dict): The metrics to analyze
        :param losses (dict): The losses to use for the backpropagation
        :param optimizer: The optimizer to use for the backpropagation
        :param num_epochs (int): Number of epochs for the training
        :param initial_epoch (int): The index of the initial epoch
        :param batch_size (int): Size a minibatch
        :param shuffle_method (Flag): DEEP_SHUFFLE flag, method of shuffling to use
        :param num_workers (int): Number of processes / threads to use for data loading
        :param verbose (int): DEEP_VERBOSE flag, How verbose the Trainer is
        :param memorize (int): DEEP_MEMORIZE flag, what data to save
        :param save_condition (int): DEEP_SAVE flag, when to save the results
        :param tester (Tester): The tester to use for validation
        :param model_name (str): The name of the model

        RETURN:
        -------

        :return: None
        """

        self.dataset = dataset
        self.name = name
        self.model = model
        self.losses = losses
        self.optimizer = optimizer
        self.metrics = Metrics() if metrics is None else metrics
        self.transform_manager = transform_manager
        self.initial_epoch = initial_epoch
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.verbose = verbose
        self.epoch = None
        self.num_minibatches = ceil(len(dataset) / batch_size)
        self.tester = tester
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        self.shuffle_method = get_corresponding_flag(
            DEEP_LIST_SHUFFLE, shuffle_method,
            fatal=False,
            default=DEEP_SHUFFLE_NONE
        )
        # self.stopping = Stopping(stopping_parameters)

        # Connect signals
        Thalamus().connect(
            receiver=self.saving_required,
            event=DEEP_EVENT_SAVING_REQUIRED,
            expected_arguments=["saving_required"]
        )
        Thalamus().connect(
            receiver=self.send_save_params,
            event=DEEP_EVENT_REQUEST_SAVE_PARAMS_FROM_TRAINER,
            expected_arguments=[]
        )

    def __evaluate_epoch(self):
        """
        AUTHORS:
        --------

        :author: Samuel Westlake, Alix Leroy

        DESCRIPTION:
        ------------

        Evaluate the model using the tester

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: The total_loss, the individual losses and the individual metrics
        """
        if self.tester is not None:
            loss, losses, metrics = self.tester.evaluate()
            Thalamus().add_signal(
                Signal(
                    event=DEEP_EVENT_ON_VALIDATION_END,
                    args={
                        "epoch_index": self.epoch,
                        "loss": loss,
                        "losses": losses,
                        "metrics": metrics,
                    }
                )
            )

    def __train(self, first_training: bool = True) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Loop over the dataset to train the network

        PARAMETERS:
        -----------

        :param first_training (bool): Whether more epochs have been required after initial training or not

        RETURN:
        -------

        :return: None
        """
        if first_training is True:
            Thalamus().add_signal(signal=Signal(event=DEEP_EVENT_ON_TRAINING_START, args={}))

        for self.epoch in range(self.initial_epoch + 1, self.num_epochs + 1):
            Thalamus().add_signal(
                signal=Signal(
                    event=DEEP_EVENT_ON_EPOCH_START,
                    args={
                        "epoch_index": self.epoch,
                        "num_epochs": self.num_epochs
                    }
                )
            )

            # Shuffle the data if required
            if self.shuffle_method is not None:
                self.dataset.shuffle(self.shuffle_method)

            # Put model into train mode and reset training losses and metrics
            self.model.train()
            self.losses.reset(self.dataset.type)
            self.metrics.reset(self.dataset.type)

            for minibatch_index, minibatch in enumerate(self.dataloader, 1):

                # Clean the given data
                inputs, labels, additional_data = self.clean_single_element_list(minibatch)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Set the data to the corresponding device
                inputs = self.to_device(inputs, self.model.device)
                labels = self.to_device(labels, self.model.device)
                additional_data = self.to_device(additional_data, self.model.device)

                # Infer the output of the batch
                outputs = self.model(*inputs)

                # Calculate train losses and total training loss
                train_loss, train_losses = self.losses.forward(
                    self.dataset.type, outputs, labels, inputs, additional_data
                )

                # Accumulates the gradient (by addition) for each parameter
                train_loss.backward()

                # Performs a parameter update based on the current gradient
                self.optimizer.step()

                # Detach output tensors (recursive)
                outputs = self.detach(outputs)

                # Transform outputs
                if self.transform_manager is not None:
                    outputs = self.transform_manager.transform(outputs, inputs, labels, additional_data)

                # Update training metrics and get values for current batch
                train_metrics = self.metrics.forward(
                    self.dataset.type, outputs, labels, inputs, additional_data
                )

                # Send signal batch end - print and save to batch history
                Thalamus().add_signal(
                    Signal(
                        event=DEEP_EVENT_ON_BATCH_END,
                        args={
                            "batch_index": minibatch_index,
                            "num_batches": self.num_minibatches,
                            "epoch_index": self.epoch,
                            "loss": train_loss.item(),
                            "losses": train_losses,
                            "metrics": train_metrics
                        }
                    )
                )

            # EPOCH END
            self.dataset.reset()  # Reset the dataset (transforms cache)

            train_loss, train_losses = self.losses.reduce(self.dataset.type)
            train_metrics = self.metrics.reduce(self.dataset.type)

            # Send signal batch end - print and save to training epoch history
            Thalamus().add_signal(
                Signal(
                    event=DEEP_EVENT_ON_EPOCH_END,
                    args={
                        "epoch_index": self.epoch,
                        "loss": train_loss.item(),
                        "losses": train_losses,
                        "metrics": train_metrics
                    }
                )
            )

            self.__evaluate_epoch()

        Thalamus().add_signal(
            Signal(
                event=DEEP_EVENT_ON_TRAINING_END,
                args={"model": self.model}
            )
        )
        self.transform_manager.finish()

    def train(self, first_training: bool = True) -> bool:
        """
        AUTHORS:
        --------

        :author: Samuel Westlake, Alix Leroy

        DESCRIPTION:
        ------------

        Fit the model to the dataset

        PARAMETERS:
        -----------

        :param first_training: (bool, optional): Whether it is the first training on the model or not

        RETURN:
        -------

        :return: None
        """
        # Pre-training checks
        if self.model is None:
            Notification(DEEP_NOTIF_ERROR, "Could not begin training : No model detected by the trainer")
            return False
        if self.losses is None:
            Notification(DEEP_NOTIF_ERROR, "Could not begin training : No losses detected by the trainer")
            return False
        if self.optimizer is None:
            Notification(DEEP_NOTIF_ERROR, "Could not begin training : No optimizer detected by the trainer")
            return False
        if self.model is None:
            Notification(DEEP_NOTIF_ERROR, "Could not begin training : No model detected by the trainer")
            return False

        Notification(DEEP_NOTIF_INFO, DEEP_MSG_TRAINING_STARTED)
        self.__train(first_training=first_training)
        Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_TRAINING_FINISHED)
        return True

    def continue_training(self, epochs=None):
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Function to know the number of epochs when continuing the training

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        if epochs is None:
            # If the user wants to continue the training, ask the number of epochs
            while True:
                epochs = Notification(DEEP_NOTIF_INPUT, "Number of epochs ? ").get()
                try:
                    epochs = int(epochs)
                    break
                except ValueError:
                    Notification(DEEP_NOTIF_WARNING, "Number of epochs must be an integer").get()
        if epochs > 0:
            self.initial_epoch = self.num_epochs
            self.num_epochs += epochs
            # Resume the training
            self.fit(first_training=False)

    def saving_required(self, saving_required: bool):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Signal to send the model to be saved if require
        NB : Contains a signal, cannot be static

        PARAMETERS:
        -----------

        :param saving_required: (bool): Whether saving the model is required or not

        RETURN:
        -------

        None
        """
        if saving_required is True:
            Thalamus().add_signal(
                signal=Signal(
                    event=DEEP_EVENT_SAVE_MODEL,
                    args={}
                )
            )

    def send_save_params(self, inp=None) -> None:
        """
        AUTHORS:
        --------

        :author: Samuel Westlake
        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Send the saving parameters to the Saver

        PARAMETERS:
        -----------

        :param inp: The input size of the model (required for ONNX models)

        RETURN:
        -------

        :return: None
        """
        Thalamus().add_signal(
            Signal(
                event=DEEP_EVENT_SEND_SAVE_PARAMS_FROM_TRAINER,
                args={"model": self.model,
                      "optimizer": self.optimizer,
                      "epoch_index": self.epoch,
                      "validation_loss": self.validation_loss,
                      "inp": inp}
            )
        )

    @staticmethod
    def clean_single_element_list(batch: list) -> list:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Convert single element lists from the batch into an element

        PARAMETERS:
        -----------

        :param batch->list: The batch to clean

        RETURN:
        -------

        :return cleaned_batch->list: The cleaned batch
        """
        cleaned_minibatch = []
        # For each entry in the minibatch:
        # If it is a single element list -> Make it the single element
        # If it is an empty list -> Make it None
        # Else -> Do not change
        for item in batch:
            if isinstance(item, list) and len(item) == 1:
                cleaned_minibatch.append(item[0])
            elif isinstance(item, list) and len(item) == 0:
                cleaned_minibatch.append(None)
            else:
                cleaned_minibatch.append(item)
        return cleaned_minibatch

    def to_device(self, x, device):
        if isinstance(x, list):
            x_ = []
            for item in x:
                item = self.to_device(item, device)
                x_.append(item)
            return x_

        else:
            try:
                return x.to(device)
            except AttributeError:
                try:
                    return [item.to(device=device) for item in x if item is not None]
                except TypeError:
                    return None

    def detach(self, x):
        if isinstance(x, list):
            x = [self.detach(item) for item in x]
        elif isinstance(x, tuple):
            x = tuple([self.detach(item) for item in x])
        elif isinstance(x, dict):
            x = {key: self.detach(item) for key, item in x.items()}
        elif isinstance(x, Namespace):
            x = {key: self.detach(item) for key, item in x.__dict__.items()}
        else:
            try:
                x = x.detach()
            except AttributeError:
                pass
        return x
