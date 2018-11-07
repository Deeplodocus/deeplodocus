import torch
import torch.onnx as onnx
from torch.nn import Module
import os
import __main__

from deeplodocus.utils.notification import Notification
from deeplodocus.utils.end import End
from deeplodocus.utils.flags import *


class Saver(object):


    def __init__(self,
                 save_condition:int = DEEP_SAVE_CONDITION_AUTO,
                 save_model_method = DEEP_SAVE_NET_FORMAT_PYTORCH,
                 overwatch_metric = "total_loss",
                 write_logs:bool=True):

        self.write_logs=write_logs
        self.save_model_method = save_model_method
        self.save_condition = save_condition
        self.overwatch_metric = overwatch_metric
        self.path = os.path.dirname(__main__.__file__)+ "/results/models"


    """
    ON BATCH END NOT TO BE IMPLEMENTED FOR EFFICIENCY REASONS
    def on_batch_end(self, model:Module):
        pass
    """

    def on_epoch_end(self, model:Module)->None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Called at each ended epoch

        PARAMETERS:
        -----------

        :param model->torch.nn.Module: The model to be saved if required

        RETURN:
        -------

        :return: None
        """

        # If we want to save the model at each epoch
        if self.save_condition == DEEP_SAVE_CONDITION_END_EPOCH:
            self.__save_model(model)

        # If we want to save the model only if we had an improvement over a metric
        elif self.save_condition == DEEP_SAVE_CONDITION_AUTO:
            if self.__is_saving_required() is True:
                self.__save_model(model)

    def on_training_end(self, model:Module)->None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Called once the training is finished

        PARAMETERS:
        -----------

        :param model->torch.nn.Module: The model to be saved if required

        RETURN:
        -------

        :return: None
        """
        if self.save_condition == DEEP_SAVE_CONDITION_END_TRAINING:
            self.__save_model(model)



    def __is_saving_required(self):
        return False




    def __save_model(self, model:Module)->None:

        # If we want to save to the pytorch format
        if self.save_model_method == DEEP_SAVE_NET_FORMAT_PYTORCH:
            try:
                torch.save(model.state_dict(), self.path)
            except:
                Notification(DEEP_NOTIF_ERROR, "Error while saving the pytorch model and weights" ,write_logs=self.write_logs)
                self.__handle_error_saving(model)

        # If we want to save to the ONNX format
        elif self.save_model_method == DEEP_SAVE_NET_FORMAT_ONNX:
            try:
                torch.onnx.export(model, dummy_input, "alexnet.proto", verbose=True, input_names=input_names, output_names=output_names)
            except:
                Notification(DEEP_NOTIF_ERROR, "Error while saving the ONNX model and weights" ,write_logs=self.write_logs)
                self.__handle_error_saving(model)

        Notification(DEEP_NOTIF_SUCCESS, "Model and weights saved", write_logs=self.write_logs)

    def __handle_error_saving(self, model:Module)->None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

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
        Notification(DEEP_NOTIF_ERROR, "Please make sure you have the permission to write for this following file : " + str(self.save_path), write_logs=self.write_logs)
        response = ""

        while response.lower() != ("y" or "n"):
            response = Notification(DEEP_NOTIF_INPUT,
                                    "Would you try to try again to save? (y/n)",
                                    write_logs=self.write_logs).get()

        if response.lower() == "y":
            self.__save_model(model)
        else:
            response = ""

            while response.lower() != ("y" or "n"):
                response = Notification(DEEP_NOTIF_INPUT, "Would you like to save in another format, if not Deeplodocus will be closed ? (y/n)" ,write_logs=self.write_logs).get()

            if response.lower() == "n":
                response = ""

                while response.lower() != ("y" or "n"):
                    Notification(DEEP_NOTIF_WARNING, "You will lose all your data if Deeplodocus is closed !" ,write_logs=self.write_logs)
                    response = Notification(DEEP_NOTIF_INPUT, "Are you sure to close Deeplodocus (y/n)", write_logs=self.write_logs).get()

                if response.lower() == "n":
                    self.__handle_error_saving()
                else:
                    End(error=False) #Exiting the program
            else:
                response = ""

                while response.lower() != ("pytorch" or "onnx"):
                    response = Notification(DEEP_NOTIF_INPUT, "What format would you like to save ? (pytorch/onnx)", write_logs=self.write_logs).get()

                if response.lower() == "pytorch":
                    self.save_model_method = DEEP_SAVE_NET_FORMAT_PYTORCH
                elif response.lower() == "onnx":
                    self.save_model_method = DEEP_SAVE_NET_FORMAT_ONNX

                self.__save_model(model)

