from decimal import Decimal
from math import ceil
from typing import Union
from torch.utils.data import DataLoader

from deeplodocus.brain.signal import Signal
from deeplodocus.brain.thalamus import Thalamus
from deeplodocus.core.metrics import Losses, Metrics
from deeplodocus.data.load.dataset import Dataset
from deeplodocus.flags import *
from deeplodocus.utils.generic_utils import get_corresponding_flag, ProgressBar
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
            shuffle: Flag = DEEP_SHUFFLE_NONE,
            name: str = "Inferer"
    ):
        self.dataset = dataset
        self.model = model
        self.transform_manager = transform_manager
        self.losses = losses
        self.metrics = Metrics() if metrics is None else metrics
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.name = name
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

    @staticmethod
    def compose_text(loss: float, losses: dict, metrics: dict, sep: str = " : "):
        return sep.join(
            ["%s : %.4e" % (DEEP_LOG_TOTAL_LOSS.name, Decimal(loss))]
            + ["%s : %.4e" % (loss_name, Decimal(value)) for (loss_name, value) in losses.items()]
            + ["%s : %.4e " % (metric_name, Decimal(value)) for (metric_name, value) in metrics.items()]
        )


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
            shuffle: Flag = DEEP_SHUFFLE_NONE,
            name: str = "Tester"
    ):
        super(Tester, self).__init__(
            dataset, model, transform_manager, losses,
            metrics=metrics,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            name=name
        )
        self.progress_bar = None

    def evaluate(self, silent: bool = False, progress_bar: Union[ProgressBar, bool] = True, prefix: str = "Evaluation :"):
        self.evaluation_start(silent=silent, progress_bar=progress_bar, prefix=prefix)
        for batch in self.dataloader:
            self.evaluation_batch(batch)
        return self.evaluation_end(silent=silent)

    def evaluation_start(self, silent: bool = False, progress_bar: bool = False, prefix: str = "Evaluation :"):
        if progress_bar is True:
            self.progress_bar = ProgressBar(self.get_num_batches(), prefix=prefix)
        elif isinstance(progress_bar, ProgressBar):
            self.progress_bar = progress_bar
        if not silent:
            Notification(DEEP_NOTIF_INFO, DEEP_MSG_EVALUATION_STARTED)
        self.model.eval()  # Put model into evaluation mode
        self.losses.reset(self.dataset.type)  # Reset corresponding losses
        self.metrics.reset(self.dataset.type)  # Reset corresponding metrics

    def evaluation_end(self, silent: bool = False):
        self.progress_bar = None
        self.transform_manager.finish()  # Call finish on all output transforms
        loss, losses = self.losses.reduce(self.dataset.type)  # Get total loss and mean of each loss
        metrics = self.metrics.reduce(self.dataset.type)  # Get total metric values
        if not silent:
            Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_EVALUATION_FINISHED)
            Notification(DEEP_NOTIF_RESULT, self.compose_text(loss, losses, metrics))
        return loss, losses, metrics

    def evaluation_batch(self, batch):
        inputs, labels, additional_data = self.clean_single_element_list(batch)
        inputs = self.to_device(inputs, self.model.device)  # Send data to device
        labels = self.to_device(labels, self.model.device)
        additional_data = self.to_device(labels, self.model.device)
        with torch.no_grad():
            outputs = self.model(*inputs)  # Infer the outputs from the model over the given mini batch
        outputs = self.detach(outputs)  # Detach the tensor from the graph
        self.losses.forward(self.dataset.type, outputs, labels, inputs, additional_data)  # Compute losses
        outputs = self.transform_manager.transform(outputs, inputs, labels, additional_data)  # Output transforms
        self.metrics.forward(self.dataset.type, outputs, labels, inputs, additional_data)  # Compute metrics
        if self.progress_bar is not None:
            self.progress_bar.step()


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
            batch_size: int = 32,
            num_workers: int = 1,
            shuffle: Flag = DEEP_SHUFFLE_NONE,
            name: str = "Trainer",
            verbose: Flag = DEEP_VERBOSE_BATCH,
            validator: Union[Tester, None] = None,
    ):
        super(Trainer, self).__init__(
            dataset, model, transform_manager, losses,
            metrics=metrics,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            name=name
        )
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.initial_epoch = initial_epoch
        self.epoch = initial_epoch
        self.verbose = verbose
        self.validator = validator
        self.batch_index = 1
        self.train_loss = None
        self.train_losses = None
        self.train_metrics = None
        self.val_loss = None
        self.val_losses = None
        self.val_metrics = None

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
        # Infer initial epoch
        self.initial_epoch = self.initial_epoch if initial_epoch is None else initial_epoch
        if self.initial_epoch is None:
            try:
                self.initial_epoch = self.model.num_epochs
            except AttributeError:
                self.initial_epoch = 0
        # Update num_epochs
        self.num_epochs = self.num_epochs if num_epochs is None else num_epochs
        self.num_epochs += self.initial_epoch
        # Go
        self.training_start()
        for self.epoch in range(self.initial_epoch + 1, self.num_epochs + 1):
            self.epoch_start()
            for self.batch_index, batch in enumerate(self.dataloader, 1):
                self.batch_start(batch)
            self.epoch_end()
        self.training_end()

    def evaluate(self):
        if self.validator is not None:
            loss, losses, metrics = self.validator.evaluate(
                silent=True,
                progress_bar=self.progress_bar if DEEP_VERBOSE_TRAINING.corresponds(self.verbose) else True,
                prefix="Validation :"
            )
            self.send_validation_end_signal(
                epoch_index=self.epoch,
                loss=loss,
                losses=losses,
                metrics=metrics
            )
            return loss, losses, metrics

    def training_start(self):
        # Initialise training progress bar if one is required
        if DEEP_VERBOSE_TRAINING.corresponds(self.verbose):
            n = (self.num_epochs - self.initial_epoch) * (self.get_num_batches() + self.validator.get_num_batches())
            self.progress_bar = ProgressBar(total=n, prefix="")
        Notification(DEEP_NOTIF_INFO, DEEP_MSG_TRAINING_STARTED)
        self.send_training_start_signal()

    def epoch_start(self):
        # Initialise epoch progress bar if one is required
        if DEEP_VERBOSE_EPOCH.corresponds(self.verbose):
            self.progress_bar = ProgressBar(total=self.get_num_batches(), prefix="Training :  ")
        elif DEEP_VERBOSE_TRAINING.corresponds(self.verbose):
            self.update_prefix(fill=4)
        self.send_epoch_start_signal(epoch_index=self.epoch, num_epochs=self.num_epochs)
        v = DEEP_VERBOSE_BATCH.corresponds(self.verbose) or DEEP_VERBOSE_EPOCH.corresponds(self.verbose)
        if v:
            Notification(DEEP_NOTIF_INFO, DEEP_MSG_EPOCH_START % self.epoch)
        self.dataset.shuffle(self.shuffle, verbose=v)  # Shuffle dataset
        self.model.train()  # Put model into train mode
        self.losses.reset(self.dataset.type)  # Reset training losses
        self.metrics.reset(self.dataset.type)  # Reset training metrics

    def batch_start(self, batch):
        inputs, labels, additional_data = self.clean_single_element_list(batch)  # Clean the given data
        self.optimizer.zero_grad()  # zero the parameter gradients
        inputs = self.to_device(inputs, self.model.device)  # Send data to device
        labels = self.to_device(labels, self.model.device)
        additional_data = self.to_device(additional_data, self.model.device)
        outputs = self.model(*inputs)  # Forward pass
        loss, losses = self.losses.forward(
            self.dataset.type, outputs, labels, inputs, additional_data
        )  # Calculate train losses and total training loss
        loss.backward()  # Backward pass
        self.optimizer.step()
        outputs = self.detach(outputs)  # Detach output tensors
        outputs = self.transform_manager.transform(outputs, inputs, labels, additional_data)
        metrics = self.metrics.forward(self.dataset.type, outputs, labels, inputs, additional_data)
        if DEEP_VERBOSE_BATCH.corresponds(self.verbose):
            self.print_batch(loss.item(), losses, metrics)
        else:
            self.progress_bar.step()
        self.send_batch_end_signal(
            batch_index=self.batch_index,
            num_batches=self.get_num_batches(),
            epoch_index=self.epoch,
            loss=loss.item(),
            losses=losses,
            metrics=metrics
        )

    def epoch_end(self):
        self.dataset.reset()  # Reset the dataset (transforms cache)
        self.train_loss, self.train_losses = self.losses.reduce(self.dataset.type)  # Calculate total loss values
        self.train_metrics = self.metrics.reduce(self.dataset.type)  # Calculate total metric values
        if not DEEP_VERBOSE_TRAINING.corresponds(self.verbose):
            self.print_epoch()  # Print training epoch results
        self.send_epoch_end_signal()
        self.transform_manager.finish()  # Call finish method on output transforms
        self.val_loss, self.val_losses, self.val_metrics = self.evaluate()  # Validate
        if not DEEP_VERBOSE_TRAINING.corresponds(self.verbose):
            self.print_validation()
        if not DEEP_VERBOSE_TRAINING.corresponds(self.verbose):  # Print epoch end
            Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_EPOCH_END % self.epoch)

    def training_end(self):
        if DEEP_VERBOSE_TRAINING.corresponds(self.verbose):
            self.print_epoch()
            self.print_validation()
        self.send_training_end_signal()
        self.initial_epoch = self.epoch
        Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_TRAINING_FINISHED)

    def print_batch(self, loss, losses, metrics):
        Notification(
            DEEP_NOTIF_RESULT, "[%i : %i/%i] : %s" % (
                self.epoch,
                self.batch_index,
                self.get_num_batches(),
                self.compose_text(loss, losses, metrics)
            )
        )

    def print_epoch(self):
        Notification(
            DEEP_NOTIF_RESULT,
            "Epoch %i : %s : %s" % (
                self.epoch, TRAINING, self.compose_text(self.train_loss, self.train_losses, self.train_metrics)
            )
        )

    def print_validation(self):
        Notification(
            DEEP_NOTIF_RESULT,
            "Epoch %i : %s : %s" % (
                self.epoch, VALIDATION, self.compose_text(self.val_loss, self.val_losses, self.val_metrics)
            )
        )

    def update_prefix(self, fill: int = 4):
        self.progress_bar.prefix = "Epoch : %s" % str(self.epoch).rjust(fill, " ")

    @staticmethod
    def send_batch_start_signal(**kwargs):
        Thalamus().add_signal(signal=Signal(event=DEEP_EVENT_BATCH_START, args=kwargs))

    @staticmethod
    def send_batch_end_signal(**kwargs):
        Thalamus().add_signal(signal=Signal(event=DEEP_EVENT_BATCH_END, args=kwargs))

    @staticmethod
    def send_epoch_start_signal(**kwargs):
        Thalamus().add_signal(signal=Signal(event=DEEP_EVENT_EPOCH_START, args=kwargs))

    def send_epoch_end_signal(self, **kwargs):
        kwargs["epoch_index"] = self.epoch
        kwargs["loss"] = self.train_loss
        kwargs["losses"] = self.train_losses
        kwargs["metrics"] = self.train_metrics
        Thalamus().add_signal(signal=Signal(event=DEEP_EVENT_EPOCH_END, args=kwargs))

    @staticmethod
    def send_training_start_signal(**kwargs):
        Thalamus().add_signal(signal=Signal(event=DEEP_EVENT_TRAINING_START, args=kwargs))

    @staticmethod
    def send_training_end_signal(**kwargs):
        Thalamus().add_signal(signal=Signal(event=DEEP_EVENT_TRAINING_END, args=kwargs))

    @staticmethod
    def send_validation_start_signal(**kwargs):
        Thalamus().add_signal(signal=Signal(event=DEEP_EVENT_VALIDATION_START, args=kwargs))

    @staticmethod
    def send_validation_end_signal(**kwargs):
        Thalamus().add_signal(signal=Signal(event=DEEP_EVENT_VALIDATION_END, args=kwargs))

