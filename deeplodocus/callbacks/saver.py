from decimal import Decimal
import torch
import os

from deeplodocus.utils.notification import Notification
from deeplodocus.flags import *
from deeplodocus.callbacks import OverWatch
from deeplodocus.brain.thalamus import Thalamus
from deeplodocus.utils.generic_utils import get_corresponding_flag
from deeplodocus.flags.flag_lists import DEEP_LIST_SAVE_SIGNAL, DEEP_LIST_SAVE_FORMATS


class Saver(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy
    :author: Samuel Westlake

    DESCRIPTION:
    ------------

    Class to handle the saving of the model
    """

    def __init__(
            self,
            model=None,
            optimizer=None,
            overwatch: OverWatch = None,
            save_directory: str = "weights",
            save_signal: Flag = DEEP_EVENT_EPOCH_END,
            method: Flag = DEEP_SAVE_FORMAT_PYTORCH,
            overwrite: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.directory = save_directory
        self.overwatch = overwatch
        self.overwrite = overwrite
        self.save_signal = get_corresponding_flag(DEEP_LIST_SAVE_SIGNAL, save_signal)
        self.method = get_corresponding_flag(DEEP_LIST_SAVE_FORMATS, method)      # Can be onnx or pt

        # Set the extension
        if DEEP_SAVE_FORMAT_PYTORCH.corresponds(self.method):
            self.extension = DEEP_EXT_PYTORCH
        elif DEEP_SAVE_FORMAT_ONNX.corresponds(self.method):
            self.extension = DEEP_EXT_ONNX

        # Connect to signals
        Thalamus().connect(
            receiver=self.save_model,
            event=DEEP_EVENT_SAVE_MODEL,
            expected_arguments=[]
        )

    def on_epoch_end(self, loss, losses, metrics=None) -> None:
        if self.save_signal.corresponds(DEEP_SAVE_SIGNAL_END_EPOCH):
            self.save_model()
        elif self.save_signal.corresponds(DEEP_SAVE_SIGNAL_AUTO):
            if self.overwatch.watch(DEEP_DATASET_TRAIN, loss, losses, metrics):
                self.save_model()

    def on_validation_end(self, loss, losses, metrics=None):
        if self.save_signal.corresponds(DEEP_SAVE_SIGNAL_END_EPOCH):
            self.save_model()
        elif self.save_signal.corresponds(DEEP_SAVE_SIGNAL_AUTO):
            if self.overwatch.watch(DEEP_DATASET_VAL, loss, losses, metrics):
                self.save_model()

    def on_training_end(self) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Called once the training is finished

        PARAMETERS:
        -----------


        RETURN:
        -------

        :return: None
        """
        if DEEP_SAVE_SIGNAL_END_TRAINING.corresponds(self.save_signal):
            self.save_model()

    def on_overwatch_metric_computed(self, current_overwatch_metric: OverWatch):
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Check if saving the model is required

        PARAMETERS:
        -----------

        :param current_overwatch_metric: float: The value of the metric to over watch

        RETURN:
        -------

        :return -> bool: Whether the model should be saved or not
        """

        # Save if there is no metric to compare against
        if self.best_overwatch_metric is None:
            self.best_overwatch_metric = current_overwatch_metric
            save = True
        else:
            # If the new metric has to be smaller than the best one
            if DEEP_SAVE_CONDITION_LESS.corresponds(current_overwatch_metric.get_condition()):
                # If the model improved since last batch => Save
                if self.best_overwatch_metric.get_value() > current_overwatch_metric.get_value():
                    Notification(
                        DEEP_NOTIF_SUCCESS,
                        DEEP_MSG_SAVER_IMPROVED % (
                            current_overwatch_metric.name,
                            "%.4e" % Decimal(
                                self.best_overwatch_metric.get_value()
                                - current_overwatch_metric.get_value()
                            )
                        )
                    )
                    self.best_overwatch_metric = current_overwatch_metric
                    save = True
                # No improvement => Return False
                else:
                    Notification(
                        DEEP_NOTIF_INFO,
                        DEEP_MSG_SAVER_NOT_IMPROVED % current_overwatch_metric.name
                    )
                    save = False

            # If the new metric has to be bigger than the best one (e.g. The accuracy of a classification)
            elif DEEP_SAVE_CONDITION_GREATER.corresponds(current_overwatch_metric.get_condition()):
                # If the model improved since last batch => Save
                if self.best_overwatch_metric.get_value() < current_overwatch_metric.get_value():
                    Notification(
                        DEEP_NOTIF_SUCCESS,
                        DEEP_MSG_SAVER_IMPROVED % (
                            current_overwatch_metric.name,
                            "%.4e" % Decimal(
                                current_overwatch_metric.get_value()
                                - self.best_overwatch_metric.get_value()
                            )
                        )
                    )
                    self.best_overwatch_metric = current_overwatch_metric
                    save = True
                # No improvement => Return False
                else:
                    Notification(
                        DEEP_NOTIF_INFO,
                        DEEP_MSG_SAVER_NOT_IMPROVED % current_overwatch_metric.name
                    )
                    save = False

            else:
                Notification(DEEP_NOTIF_FATAL, "The following saving condition does not exist : %s"
                             % current_overwatch_metric.get_condition())
                save = False

        if save is True:
            self.save_model()

    def save_model(self) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Save the model

        PARAMETERS:
        -----------

        RETURN:
        -------

        :return: None
        """
        # Create the target directory
        if not os.path.isfile(self.directory):
            os.makedirs(self.directory, exist_ok=True)
        if self.overwrite:
            file_path = "%s/%s" % (self.directory, self.model.name)
        else:
            file_path = "%s/%s_%s" % (self.directory, self.model.name, str(self.model.epoch).zfill(4))
        if DEEP_SAVE_FORMAT_PYTORCH.corresponds(self.method):  # Pytorch format
            file_path += DEEP_EXT_PYTORCH
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": self.model.epoch if "epoch" in vars(self.model).keys() else None
                },
                file_path
            )
            Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_MODEL_SAVED % file_path)
        elif DEEP_SAVE_FORMAT_ONNX.corresponds(self.method):  # ONNX format
            # TODO: SAVE TO ONIX FILE FORMAT
            file_path += DEEP_EXT_ONNX
            Notification(DEEP_NOTIF_FATAL, "Save as onnx format not implemented yet")
