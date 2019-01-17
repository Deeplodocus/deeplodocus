import torch
from torch.nn import Module
import os

from deeplodocus.utils.notification import Notification
from deeplodocus.utils.flags.save import *
from deeplodocus.utils.flags.event import *
from deeplodocus.utils.flags.notif import *
from deeplodocus.utils.flags.ext import DEEP_EXT_PYTORCH, DEEP_EXT_ONNX
from deeplodocus.utils.flags.path import DEEP_PATH_SAVE_MODEL
from deeplodocus.utils.flags.msg import DEEP_MSG_MODEL_SAVED
from deeplodocus.core.metrics.over_watch_metric import OverWatchMetric
from deeplodocus.brain.signal import Signal
from deeplodocus.brain.thalamus import Thalamus


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

    def __init__(self,
                 name: str = "no__model_name",
                 save_directory: str = DEEP_PATH_SAVE_MODEL,
                 save_condition: Flag = DEEP_SAVE_CONDITION_LESS,
                 save_format: Flag = DEEP_SAVE_FORMAT_PYTORCH):

        self.save_format = save_format      # Can be onnx or pt
        self.save_condition = save_condition
        self.directory = save_directory
        self.name = name
        self.best_overwatch_metric = None

        # Set the extension
        if DEEP_SAVE_FORMAT_PYTORCH.corresponds(self.save_format):
            self.extension = DEEP_EXT_PYTORCH
        elif DEEP_SAVE_FORMAT_ONNX.corresponds(self.save_format):
            self.extension = DEEP_EXT_ONNX

        if not os.path.isfile(self.directory):
            os.makedirs(self.directory, exist_ok=True)

        # Connect the save to the computation of the overwatched metric
        Thalamus().connect(receiver=self.is_saving_required,
                           event=DEEP_EVENT_OVERWATCH_METRIC_COMPUTED,
                           expected_arguments=["current_overwatch_metric"])
        Thalamus().connect(receiver=self.on_training_end,
                           event=DEEP_EVENT_ON_TRAINING_END,
                           expected_arguments=["model"])
        Thalamus().connect(receiver=self.save_model,
                           event=DEEP_EVENT_SAVE_MODEL,
                           expected_arguments=["model"])

    """
    ON BATCH END NOT TO BE IMPLEMENTED FOR EFFICIENCY REASONS
    def on_batch_end(self, model:Module):
        pass
    """

    def on_overwatch_metric_computed(self, model: Module, current_overwatch_metric: OverWatchMetric) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Called at each ended epoch

        PARAMETERS:
        -----------

        :param model: torch.nn.Module: The model to be saved if required
        :param current_overwatch_metric: OverWatchMetric

        RETURN:
        -------

        :return: None
        """

        # If we want to save the model at each epoch
        if DEEP_SAVE_SIGNAL_END_EPOCH.corresponds(self.save_condition):
            self.save_model(model)

        # If we want to save the model only if we had an improvement over a metric
        elif DEEP_SAVE_SIGNAL_AUTO.corresponds(self.save_condition):
            if self.is_saving_required(current_overwatch_metric=current_overwatch_metric) is True:
                self.save_model(model)

    def on_training_end(self, model: Module) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Called once the training is finished

        PARAMETERS:
        -----------

        :param model: torch.nn.Module: The model to be saved if required

        RETURN:
        -------

        :return: None
        """
        if DEEP_SAVE_SIGNAL_END_TRAINING.corresponds(self.save_condition):
            self.save_model(model)

    def is_saving_required(self, current_overwatch_metric: OverWatchMetric) -> bool:
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
        save = False

        # Do not save at the first epoch
        if self.best_overwatch_metric is None:
            self.best_overwatch_metric = current_overwatch_metric
            save = False

        # If  the new metric has to be smaller than the best one
        if DEEP_SAVE_CONDITION_LESS.corresponds(current_overwatch_metric.get_condition()):
            # If the model improved since last batch => Save
            if self.best_overwatch_metric.get_value() > current_overwatch_metric.get_value():
                self.best_overwatch_metric = current_overwatch_metric
                save = True

            # No improvement => Return False
            else:
                save = False

        # If the new metric has to be bigger than the best one (e.g. The accuracy of a classification)
        elif DEEP_SAVE_CONDITION_GREATER.corresponds(current_overwatch_metric.get_condition()):
            # If the model improved since last batch => Save
            if self.best_overwatch_metric.get_value() < current_overwatch_metric.get_value():
                self.best_overwatch_metric = current_overwatch_metric
                save = True

            # No improvement => Return False
            else:
                save = False

        else:
            Notification(DEEP_NOTIF_FATAL, "The following saving condition does not exist : %s"
                         % current_overwatch_metric.get_condition())

        Thalamus().add_signal(signal=Signal(event=DEEP_EVENT_SAVING_REQUIRED, args={"saving_required": save}))
        return save

    def save_model(self, model: Module, inp=None) -> None:
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

        :param model: The model to save
        :param inp: ###

        RETURN:
        -------

        :return: None
        """

        file_path = "%s/%s%s" % (self.directory, self.name, self.extension)

        # If we want to save to the pytorch format
        if DEEP_SAVE_FORMAT_PYTORCH.corresponds(self.save_format):
            # TODO: Finish try except statements here after testing...
            # try:
            torch.save(torch.save({"epoch": epoch,
                                   "model_state_dict": model.state_dict(),
                                   "optimizer_state_dict": optimizer.state_dict(),
                                   "loss": loss}, file_path))
            # except:
            #     Notification(DEEP_NOTIF_ERROR, "Error while saving the pytorch model and weights" )
            #     self.__handle_error_saving(model)

        # If we want to save to the ONNX format
        elif DEEP_SAVE_FORMAT_ONNX.corresponds(self.save_format):
            # TODO: and here. Also fix onnx export function
            Notification(DEEP_NOTIF_FATAL, "Save as onnx format not implemented yet")
            # try:
            # torch.onnx._export(model, inp, filepath,
            #                    export_params=True,
            #                    verbose=True,
            #                    input_names=input_names,
            #                    output_names=output_names)
            # except:
            #     Notification(DEEP_NOTIF_ERROR, "Error while saving the ONNX model and weights" )
            #     self.__handle_error_saving(model)

        Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_MODEL_SAVED % file_path)

    def __handle_error_saving(self, name: str, model: Module)->None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Handle the error.
        Suggest solutions:
            - Retry to save the model
            - Change the save format
        Exit the program if no solution found

        :param model->Module: The model to save

        RETURN:
        -------

        :return: None
        """
        Notification(DEEP_NOTIF_ERROR, "Please make sure you have the permission to write for this following file : " + str(name))
        response = ""

        while response.lower() != ("y" or "n"):
            response = Notification(DEEP_NOTIF_INPUT,
                                    "Would you try to try again to save? (y/n)").get()

        if response.lower() == "y":
            self.save_model(model)
        else:
            response = ""

            while response.lower() != ("y" or "n"):
                response = Notification(DEEP_NOTIF_INPUT, "Would you like to save in another format, if not Deeplodocus will be closed ? (y/n)").get()

            if response.lower() == "n":
                response = ""

                while response.lower() != ("y" or "n"):
                    Notification(DEEP_NOTIF_WARNING, "You will lose all your data if Deeplodocus is closed !" )
                    response = Notification(DEEP_NOTIF_INPUT, "Are you sure to close Deeplodocus (y/n)").get()

                if response.lower() == "n":
                    self.__handle_error_saving()
                else:
                    End(error=False) #Exiting the program
            else:
                response = ""

                while response.lower() != ("pytorch" or "onnx"):
                    response = Notification(DEEP_NOTIF_INPUT, "What format would you like to save ? (pytorch/onnx)").get()

                if response.lower() == "pytorch":
                    self.save_model_method = DEEP_SAVE_NET_FORMAT_PYTORCH
                elif response.lower() == "onnx":
                    self.save_model_method = DEEP_SAVE_NET_FORMAT_ONNX

                self.save_model(model)

