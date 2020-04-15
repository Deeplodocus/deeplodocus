from math import ceil
from typing import Union
from torch.utils.data import DataLoader

from deeplodocus.brain.signal import Signal
from deeplodocus.brain.thalamus import Thalamus
from deeplodocus.core.metrics import Losses, Metrics
from deeplodocus.data.load.dataset import Dataset
from deeplodocus.flags import *
from deeplodocus.utils.generic_utils import get_corresponding_flag
from deeplodocus.utils.namespace import Namespace
from deeplodocus.utils.notification import Notification


class GenericInferer(object):

    def __init__(
            self,
            dataset: Dataset,
            model,
            transform_manager,
            losses: Losses,
            metrics: Union[Metrics, None] = None,
            batch_size: int = 32,
            num_workers: int = 1,
            shuffle: Flag = DEEP_SHUFFLE_NONE
    ):
        self.dataset = dataset
        self.model = model
        self.transform_manager = transform_manager
        self.losses = losses
        self.metrics = metrics if metrics is None else Metrics()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = get_corresponding_flag(
            DEEP_LIST_SHUFFLE, shuffle,
            fatal=False,
            default=DEEP_SHUFFLE_NONE
        )
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def get_num_batches(self) -> int:
        return int(ceil(len(self.dataset) / self.batch_size))

    def to_device(self, x, device):
        if isinstance(x, list):
            return [self.to_device(i, device) for i in x]
        elif isinstance(x, tuple):
            return tuple([self.to_device(i, device) for i in x])
        elif isinstance(x, dict):
            return {k: self.to_device(i, device) for k, i in x.items()}
        elif isinstance(x, Namespace):
            x.__dict__ = {k: self.to_device(i, device) for k, i in x.__dict__.items()}
            return x
        else:
            try:
                return x.to(device)
            except AttributeError:
                return x

    def detach(self, x):
        if isinstance(x, list):
            return [self.detach(item) for item in x]
        elif isinstance(x, tuple):
            return tuple([self.detach(item) for item in x])
        elif isinstance(x, dict):
            return {key: self.detach(item) for key, item in x.items()}
        elif isinstance(x, Namespace):
            x.__dict__ = {k: self.detach(i) for k, i in x.__dict__.items()}
            return x
        else:
            try:
                return x.detach()
            except AttributeError:
                return x

    @staticmethod
    def clean_single_element_list(batch: list) -> list:
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


class Tester(GenericInferer):

    def __init__(
            self,
            dataset: Dataset,
            model,
            transform_manager,
            losses: Losses,
            metrics: Union[Metrics, None] = None,
            batch_size: int = 32,
            num_workers: int = 1,
            shuffle: Flag = DEEP_SHUFFLE_NONE
    ):
        super(Tester, self).__init__(
            dataset, model, transform_manager, losses,
            metrics=metrics,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle
        )

    def evaluate(self):
        self.model.eval()  # Put model into evaluation mode
        self.losses.reset(self.dataset.type)  # Reset corresponding losses
        self.metrics.reset(self.dataset.type)  # Reset corresponding metrics
        for batch_index, batch in enumerate(self.dataloader, 0):
            inputs, labels, additional_data = self.clean_single_element_list(batch)
            inputs = self.to_device(inputs, self.model.device)  # Send data to device
            labels = self.to_device(labels, self.model.device)
            additional_data = self.to_device(labels, self.model.device)
            with torch.no_grad():
                outputs = self.model(*inputs)  # Infer the outputs from the model over the given mini batch
            outputs = self.detach(outputs)  # Detach the tensor from the graph
            self.losses.forward(self.dataset.type, outputs, labels, inputs, additional_data)  # Compute losses
            if self.transform_manager is not None:
                outputs = self.transform_manager.transform(inputs, outputs, labels, additional_data)
            self.metrics.forward(self.dataset.type, outputs, labels, inputs, additional_data)  # Compute metrics
        self.transform_manager.finish()  # Call finish on all output transforms
        loss, losses = self.losses.reduce(self.dataset.type)  # Get total loss and mean of each loss
        metrics = self.metrics.reduce(self.dataset.type)  # Get total metric values
        return loss.item(), losses, metrics


class Trainer(GenericInferer):

    def __init__(
            self,
            dataset: Dataset,
            model,
            optimizer,
            transform_manager,
            losses: Losses,
            metrics: Metrics = Union[Metrics, None],
            num_epochs: int = 1,
            initial_epoch: Union[int, None] = None,
            batch_size: int = 4,
            num_workers: int = 4,
            shuffle: Flag = DEEP_SHUFFLE_NONE,
            verbose: Flag = DEEP_VERBOSE_BATCH,
            validator: Union[Tester, None] = None
    ):
        super(Trainer, self).__init__(
            dataset, model, transform_manager, losses,
            metrics=metrics,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle
        )
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.initial_epoch = initial_epoch
        self.epoch = initial_epoch
        self.verbose = verbose
        self.validator = validator

    def train(self, num_epochs: Union[int, None] = None, initial_epoch: Union[int, None] = None):
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
        # Update num_epochs
        self.num_epochs = self.num_epochs if num_epochs is None else num_epochs
        # Infer initial epoch
        self.initial_epoch = self.initial_epoch if initial_epoch is None else initial_epoch
        if self.initial_epoch is None:
            try:
                self.initial_epoch = self.model.num_epochs
            except AttributeError:
                self.initial_epoch = 1
        # Go
        Notification(DEEP_NOTIF_INFO, DEEP_MSG_TRAINING_STARTED)
        self.__train()
        Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_TRAINING_FINISHED)

    def evaluate(self):
        if self.validator is not None:
            loss, losses, metrics = self.validator.evaluate()
            self.send_validation_end_signal(
                epoch_index=self.epoch,
                loss=loss,
                losses=losses,
                metrics=metrics
            )
            return loss, losses, metrics
        else:
            return None, None, None

    def __train(self):
        self.send_training_start_signal()
        for self.epoch in range(self.initial_epoch + 1, self.num_epochs + 1):
            self.send_epoch_start_signal(epoch_index=self.epoch, num_epochs=self.num_epochs)
            self.dataset.shuffle(self.shuffle)  # Shuffle dataset
            self.model.train()  # Put model into train mode
            self.losses.reset(self.dataset.type)  # Reset training losses
            self.metrics.reset(self.dataset.type)  # Reset training metrics
            for batch_index, batch in enumerate(self.dataloader, 1):
                inputs, labels, additional_data = self.clean_single_element_list(batch)  # Clean the given data
                self.optimizer.zero_grad()  # zero the parameter gradients
                inputs = self.to_device(inputs, self.model.device)  # Send data to device
                labels = self.to_device(labels, self.model.device)
                additional_data = self.to_device(additional_data, self.model.device)
                outputs = self.model(*inputs)  # Forward pass
                train_loss, train_losses = self.losses.forward(
                    self.dataset.type, outputs, labels, inputs, additional_data
                )  # Calculate train losses and total training loss
                train_loss.backward()  # Backward pass
                self.optimizer.step()
                outputs = self.detach(outputs)  # Detach output tensors
                # Transform outputs
                if self.transform_manager is not None:
                    outputs = self.transform_manager.transform(outputs, inputs, labels, additional_data)
                train_metrics = self.metrics.forward(
                    self.dataset.type, outputs, labels, inputs, additional_data
                )  # Update training metrics and get values for current batch
                self.send_batch_end_signal(
                    batch_inde=batch_index,
                    num_batches=self.get_num_batches(),
                    epoch_index=self.epoch,
                    loss=train_loss.item(),
                    losses=train_losses,
                    metrics=train_metrics
                )
            self.dataset.reset()  # Reset the dataset (transforms cache)
            train_loss, train_losses = self.losses.reduce(self.dataset.type)
            train_metrics = self.metrics.reduce(self.dataset.type)
            self.send_epoch_end_signal(
                epoch_index=self.epoch,
                loss=train_loss.item(),
                losses=train_losses,
                metrics=train_metrics
            )
            loss, losses, metrics = self.evaluate()
        self.transform_manager.finish()
        self.initial_epoch = self.epoch
        self.send_training_end_signal()

    @staticmethod
    def send_training_start_signal(**kwargs):
        Thalamus().add_signal(signal=Signal(event=DEEP_EVENT_ON_TRAINING_START, args=kwargs))

    @staticmethod
    def send_epoch_start_signal(**kwargs):
        Thalamus().add_signal(signal=Signal(event=DEEP_EVENT_ON_EPOCH_START, args=kwargs))

    @staticmethod
    def send_batch_end_signal(**kwargs):
        Thalamus().add_signal(signal=Signal(event=DEEP_EVENT_ON_BATCH_END, args=kwargs))

    @staticmethod
    def send_epoch_end_signal(**kwargs):
        Thalamus().add_signal(signal=Signal(event=DEEP_EVENT_ON_EPOCH_END, args=kwargs))

    @staticmethod
    def send_validation_end_signal(**kwargs):
        Thalamus().add_signal(signal=Signal(event=DEEP_EVENT_ON_EPOCH_END, args=kwargs))

    @staticmethod
    def send_training_end_signal(**kwargs):
        Thalamus().add_signal(signal=Signal(event=DEEP_EVENT_ON_TRAINING_END, args=kwargs))
