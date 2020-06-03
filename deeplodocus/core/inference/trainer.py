from typing import Union

from deeplodocus.brain.signal import Signal
from deeplodocus.brain.thalamus import Thalamus
from deeplodocus.core.metrics import Losses, Metrics
from deeplodocus.data.load.dataset import Dataset
from deeplodocus.flags import *
from deeplodocus.utils.generic_utils import ProgressBar
from deeplodocus.utils.notification import Notification
from deeplodocus.core.inference import Inferer, Tester


class Trainer(Inferer):

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
        self.verbose = verbose
        self.validator = validator
        self.initial_epoch = initial_epoch
        self.epoch = None
        self.batch_index = 1
        self.train_loss = None
        self.train_losses = None
        self.train_metrics = None
        self.val_loss = None
        self.val_losses = None
        self.val_metrics = None
        self.progress_bar = None

    def train(self, num_epochs: Union[int, None] = None):
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
        if self.initial_epoch is None:
            self.initial_epoch = self.model.epoch if "epoch" in vars(self.model).keys() else 0

        # Go
        self.training_start()
        for self.epoch in range(self.initial_epoch + 1, self.num_epochs + self.initial_epoch + 1):
            self.epoch_start()
            for self.batch_index, batch in enumerate(self.dataloader, 1):
                self.forward(batch)
            self.epoch_end()
        self.training_end()

    def evaluate(self):
        if self.validator is not None:
            self.val_loss, self.val_losses, self.val_metrics = self.validator.evaluate(
                silent=True,
                progress_bar=self.progress_bar if DEEP_VERBOSE_TRAINING.corresponds(self.verbose) else True,
                prefix="Epoch %s : Validation :" % str(self.epoch).rjust(4)
            )  # Evaluate
            if not DEEP_VERBOSE_TRAINING.corresponds(self.verbose):
                self.print_validation()  # Print validation results
            self.send_validation_end_signal(
                epoch_index=self.epoch,
                loss=self.val_loss,
                losses=self.val_losses,
                metrics=self.val_metrics
            )  # Signal to Hippocampus (History and Saver)

    def training_start(self):
        # Initialise training progress bar if one is required
        if DEEP_VERBOSE_TRAINING.corresponds(self.verbose):
            n = (self.num_epochs - self.initial_epoch) * (self.get_num_batches() + self.validator.get_num_batches())
            self.progress_bar = ProgressBar(
                total=n,
                prefix="DEEP PROGRESS : Epoch %s :" % str(self.epoch).rjust(4)
            )
        Notification(DEEP_NOTIF_INFO, DEEP_MSG_TRAINING_STARTED)
        self.send_training_start_signal()

    def epoch_start(self):
        # Initialise epoch progress bar if one is required
        if DEEP_VERBOSE_EPOCH.corresponds(self.verbose):
            self.progress_bar = ProgressBar(
                total=self.get_num_batches(),
                prefix="DEEP PROGRESS : Epoch %s : Training   :" % str(self.epoch).rjust(4)
            )
        elif DEEP_VERBOSE_TRAINING.corresponds(self.verbose):
            self.update_prefix(fill=4)
        v = DEEP_VERBOSE_BATCH.corresponds(self.verbose) or DEEP_VERBOSE_EPOCH.corresponds(self.verbose)
        if v:
            Notification(DEEP_NOTIF_INFO, DEEP_MSG_EPOCH_START % self.epoch)
        self.dataset.shuffle(self.shuffle, verbose=v)  # Shuffle dataset
        self.model.train()  # Put model into train mode
        self.losses.reset(self.dataset.type)  # Reset training losses
        self.metrics.reset(self.dataset.type)  # Reset training metrics

    def forward(self, batch):
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

    def forward2(self, batch, split=4):
        # Please leave in for now
        # Example of custom forward method - in development
        inputs, labels, additional_data = self.clean_single_element_list(batch)  # Clean the given data
        b = inputs[0].shape[0]

        if b < split:
            split = b

        self.optimizer.zero_grad()  # zero the parameter gradients
        for i in range(split):
            i0 = int(i * b / split)
            i1 = int((i + 1) * b / split)

            # Get mini mini batch and put onto device
            inp = self.to_device([item[i0: i1] for item in inputs], self.model.device)
            lab = self.to_device(labels[i0: i1], self.model.device)
            add = None if additional_data is None else self.to_device(additional_data[i0: i1], self.model.device)

            out = self.model(*inp)  # Forward pass

            mini_loss, mini_losses = self.losses.forward(self.dataset.type, out, lab, inp, add)  # Loss function
            out = self.detach(out)  # Detach output tensors
            out = self.transform_manager.transform(out, inp, lab, add)  # Output transforms
            mini_metrics = self.metrics.forward(self.dataset.type, out, lab, inp, add)  # Metrics

            mini_loss.backward()

            if i == 0:
                loss = mini_loss
                losses = {key: [item] for key, item in mini_losses.items()}
                metrics = {key: [item] for key, item in mini_metrics.items()}
            else:
                loss = loss + mini_loss
                [losses[key].append(item) for key, item in mini_losses.items()]
                [metrics[key].append(item) for key, item in mini_metrics.items()]

        # Reduce loss, losses and metrics
        loss /= split
        losses = {key: sum(item) / split for key, item in losses.items()}
        metrics = {key: vars(self.metrics)[key].reduce_method(item) for key, item in metrics.items()}

        # Update parameters
        self.optimizer.step()

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
        self.progress_bar = None  # This statement is necesssary
        self.model.epoch = self.epoch  # Make sure this happens before saver is called
        self.dataset.reset()  # Reset the dataset (transforms cache)
        self.train_loss, self.train_losses = self.losses.reduce(self.dataset.type)  # Calculate total loss values
        self.train_metrics = self.metrics.reduce(self.dataset.type)  # Calculate total metric values
        if not DEEP_VERBOSE_TRAINING.corresponds(self.verbose):
            self.print_epoch()  # Print training epoch results
        self.send_epoch_end_signal()
        self.transform_manager.finish()  # Call finish method on output transforms
        self.evaluate()  # Validate
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
            "Epoch %s : Training   : %s" % (
                str(self.epoch).rjust(4),
                self.compose_text(self.train_loss, self.train_losses, self.train_metrics)
            )
        )

    def print_validation(self):
        Notification(
            DEEP_NOTIF_RESULT,
            "Epoch %s : %s : %s" % (
                str(self.epoch).rjust(4),
                VALIDATION,
                self.compose_text(self.val_loss, self.val_losses, self.val_metrics)
            )
        )

    def update_prefix(self, fill: int = 4):
        self.progress_bar.prefix = "DEEP PROGRESS : Epoch %s :" % str(self.epoch).rjust(fill)

    @staticmethod
    def send_batch_start_signal(**kwargs):
        Thalamus().add_signal(signal=Signal(event=DEEP_EVENT_BATCH_START, args=kwargs))

    @staticmethod
    def send_batch_end_signal(**kwargs):
        Thalamus().add_signal(signal=Signal(event=DEEP_EVENT_BATCH_END, args=kwargs))

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