# Import modules from Python library
from typing import Union

# Import modules from deeplodocus
from deeplodocus.callbacks.saver import Saver
from deeplodocus.callbacks.history import History
from deeplodocus.callbacks import OverWatch
from deeplodocus.brain.thalamus import Thalamus
from deeplodocus.flags import *
from deeplodocus.utils.generic_utils import get_corresponding_flag

Num = Union[int, float]


class Hippocampus(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy
    :author: Samuel Westlake

    DESCRIPTION:
    ------------

    The Hippocampus class manages all the instances related to the information saved for short
    and long terms by Deeplodocus

    The following information are handled by the Hippocampus:
        - The history
        - The saving of the model and weights
    """

    def __init__(
            self,
            model=None,
            optimizer=None,
            overwatch: OverWatch = OverWatch(),
            save_signal: Flag = DEEP_SAVE_SIGNAL_AUTO,
            method: Flag = DEEP_SAVE_FORMAT_PYTORCH,
            overwrite: bool = False,
            enable_train_batches: bool = True,
            enable_train_epochs: bool = True,
            enable_validation: bool = True,
            history_directory: str = "history",
            weights_directory: str = "weights"
    ):
        self.name = "Memory"
        save_signal = get_corresponding_flag(DEEP_LIST_SAVE_SIGNAL, save_signal)
        self.history = History(
            log_dir=history_directory,
            save_signal=save_signal,
            enable_train_batches=enable_train_batches,
            enable_train_epochs=enable_train_epochs,
            enable_validation=enable_validation
        )
        self.saver = Saver(
            model=model,
            optimizer=optimizer,
            overwatch=overwatch,
            save_directory=weights_directory,
            save_signal=save_signal,
            method=method,
            overwrite=overwrite
        )
        self._model = model
        self._optimizer = optimizer
        self._overwatch = overwatch
        # Connect to signals
        Thalamus().connect(
            receiver=self.on_batch_end,
            event=DEEP_EVENT_BATCH_END,
            expected_arguments=["batch_index", "num_batches", "epoch_index", "loss", "losses", "metrics"]
        )
        Thalamus().connect(
            receiver=self.on_epoch_end,
            event=DEEP_EVENT_EPOCH_END,
            expected_arguments=["epoch_index", "loss", "losses", "metrics"]
        )
        Thalamus().connect(
            receiver=self.on_validation_end,
            event=DEEP_EVENT_VALIDATION_END,
            expected_arguments=["epoch_index", "loss", "losses", "metrics"]
        )
        Thalamus().connect(
            receiver=self.on_train_start,
            event=DEEP_EVENT_TRAINING_START,
            expected_arguments=[]
        )
        Thalamus().connect(
            receiver=self.on_train_end,
            event=DEEP_EVENT_TRAINING_END,
            expected_arguments=[]
        )
        Thalamus().connect(
            receiver=self.send_training_loss,
            event=DEEP_EVENT_REQUEST_TRAINING_LOSS,
            expected_arguments=[]
        )

    def on_train_start(self):
        self.history.on_train_start()

    def on_batch_end(self, *args, **kwargs):
        self.history.on_batch_end(*args, **kwargs)

    def on_epoch_end(self, epoch_index, loss, losses, metrics=None):
        self.history.on_epoch_end(epoch_index, loss, losses, metrics)
        self.saver.on_epoch_end(loss, losses, metrics)

    def on_validation_end(self, epoch_index, loss, losses, metrics=None):
        self.history.on_validation_end(epoch_index, loss, losses, metrics)
        self.saver.on_validation_end(loss, losses, metrics)

    def on_train_end(self):
        self.history.on_train_end()

    def send_training_loss(self):
        print("NOT IMPLEMENTED : HIPPOCAMPUS : SEND TRAINING LOSS")

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self.saver.model = model

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self.saver.optimizer = optimizer

    @property
    def overwatch(self):
        return self._overwatch

    @overwatch.setter
    def overwatch(self, overwatch):
        self._overwatch = overwatch
        self.saver.overwatch = overwatch

