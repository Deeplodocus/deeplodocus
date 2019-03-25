from decimal import Decimal
import torch
from torch.nn import Module
import os

from deeplodocus.utils.notification import Notification
from deeplodocus.utils.flags.save import *
from deeplodocus.utils.flags.event import *
from deeplodocus.utils.flags.notif import *
from deeplodocus.utils.flags.ext import DEEP_EXT_PYTORCH, DEEP_EXT_ONNX
from deeplodocus.utils.flags.msg import DEEP_MSG_MODEL_SAVED, DEEP_MSG_SAVER_IMPROVED, DEEP_MSG_SAVER_NOT_IMPROVED
from deeplodocus.core.metrics.over_watch_metric import OverWatchMetric
from deeplodocus.brain.signal import Signal
from deeplodocus.brain.thalamus import Thalamus
from deeplodocus.utils.generic_utils import get_corresponding_flag
from deeplodocus.utils.flags.flag_lists import DEEP_LIST_SAVE_SIGNAL, DEEP_LIST_SAVE_FORMATS


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
            name: str = "no_model_name",
            save_directory: str = "weights",
            save_signal: Flag = DEEP_EVENT_ON_EPOCH_END,
            method: Flag = DEEP_SAVE_FORMAT_PYTORCH,
            overwrite: bool = False
    ):
        self.name = name
        self.directory = save_directory
        self.save_signal = get_corresponding_flag(DEEP_LIST_SAVE_SIGNAL, save_signal)
        self.method = get_corresponding_flag(DEEP_LIST_SAVE_FORMATS, method)      # Can be onnx or pt
        self.best_overwatch_metric = None
        self.training_loss = None
        self.model = None
        self.optimizer = None
        self.epoch_index = -1
        self.batch_index = -1
        self.validation_loss = None
        self.overwrite = overwrite
        self.inp = None

        # Set the extension
        if DEEP_SAVE_FORMAT_PYTORCH.corresponds(self.method):
            self.extension = DEEP_EXT_PYTORCH
        elif DEEP_SAVE_FORMAT_ONNX.corresponds(self.method):
            self.extension = DEEP_EXT_ONNX

        if not os.path.isfile(self.directory):
            os.makedirs(self.directory, exist_ok=True)

        # Connect the save to the computation of the overwatched metric
        Thalamus().connect(
            receiver=self.on_overwatch_metric_computed,
            event=DEEP_EVENT_OVERWATCH_METRIC_COMPUTED,
            expected_arguments=["current_overwatch_metric"]
        )
        Thalamus().connect(
            receiver=self.on_training_end,
            event=DEEP_EVENT_ON_TRAINING_END,
            expected_arguments=[]
        )
        Thalamus().connect(
            receiver=self.save_model,
            event=DEEP_EVENT_SAVE_MODEL,
            expected_arguments=[]
        )
        Thalamus().connect(
            receiver=self.set_training_loss,
            event=DEEP_EVENT_SEND_TRAINING_LOSS,
            expected_arguments=["training_loss"]
        )
        Thalamus().connect(
            receiver=self.set_save_params,
            event=DEEP_EVENT_SEND_SAVE_PARAMS_FROM_TRAINER,
            expected_arguments=[
                "model",
                "optimizer",
                "epoch_index",
                "validation_loss",
                "inp"
            ]
        )

    """
    ON BATCH END NOT TO BE IMPLEMENTED FOR EFFICIENCY REASONS
    def on_batch_end(self, model:Module):
        pass
    """

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

    def on_overwatch_metric_computed(self, current_overwatch_metric: OverWatchMetric):
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
        # Set training_loss
        Thalamus().add_signal(
            Signal(
                event=DEEP_EVENT_REQUEST_TRAINING_LOSS,
                args=[]
            )
        )

        # Set model and stuff
        Thalamus().add_signal(
            Signal(
                event=DEEP_EVENT_REQUEST_SAVE_PARAMS_FROM_TRAINER,
                args=[]
            )
        )

        file_path = self.__get_file_path()

        # If we want to save to the pytorch format
        if DEEP_SAVE_FORMAT_PYTORCH.corresponds(self.method):
            # TODO: Finish try except statements here after testing...
            # try:
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "epoch": self.epoch_index,
                    "training_loss": self.training_loss,
                    "validation_loss": self.validation_loss,
                    "optimizer_state_dict": self.optimizer.state_dict()
                },
                file_path
            )
            # except:
            #     Notification(DEEP_NOTIF_ERROR, "Error while saving the pytorch model and weights" )
            #     self.__handle_error_saving(model)

        # If we want to save to the ONNX format
        elif DEEP_SAVE_FORMAT_ONNX.corresponds(self.method):
            # TODO: and here. Also fix onnx export function
            Notification(DEEP_NOTIF_FATAL, "Save as onnx format not implemented yet")
            # try:
            # torch.onnx._export(model, inp, file_path,
            #                    export_params=True,
            #                    verbose=True,
            #                    input_names=input_names,
            #                    output_names=output_names)
            # except:
            #     Notification(DEEP_NOTIF_ERROR, "Error while saving the ONNX model and weights" )
            #     self.__handle_error_saving(model)

        Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_MODEL_SAVED % file_path)

    def set_training_loss(self, training_loss):
        """
        :param training_loss:
        :return:
        """
        self.training_loss = training_loss

    def set_save_params(self, model, optimizer, epoch_index, validation_loss, inp):
        """
        :param model:
        :param optimizer:
        :param epoch_index:
        :param validation_loss:
        :param inp:
        :return:
        """
        self.model = model
        self.optimizer = optimizer
        self.epoch_index = epoch_index
        self.validation_loss = validation_loss
        self.inp = inp

    def __get_file_path(self):
        if self.save_signal.corresponds(DEEP_SAVE_SIGNAL_END_BATCH):
            # Set the file path as 'directory/name_epoch_batch.ext'
            file_path = "%s/%s_%s_%s%s" % (
                self.directory,
                self.name,
                str(self.epoch_index).zfill(3),
                str(self.batch_index).zfill(8),
                self.extension
            )
        # If saving at the end of each epoch
        else:
            # Set the file path as 'directory/name_epoch.ext'
            file_path = "%s/%s_%s%s" % (
                self.directory,
                self.name,
                str(self.epoch_index).zfill(3),
                self.extension
            )
        return file_path
