import torch
import torch.onnx as onnx


class Save(object):


    def __init__(self,
                 condition = "",
                 metric = "loss"):


        self.condition = condition     # save condition : "epoch" = on every epoch, "auto" = everytime the selected metric improves, "end" = save the model at the end of the training
        self.method = "pytorch"
        self.metric = metric

        #
        # CHECKS
        #

        # Check the select metric is correct for the saving
        if self.metric not in running_metrics[:][0]
            raise ValueError("The selected metric to overwatch for saving the model is not computed by the model.")


    def on_epoch_end(self):

        # If we want to save the model at each epoch
        if self.save_condition == "epoch":
            self.save_model()

        # If we want to save the model only if we had an improvement over a metric
        elif self.save_condition == "auto":
            is_saving_required = self.is_saving_required()

            if is_saving_required is True:
                self.save_model()

    def on_training_end(self):

        if self.save_condition == "end":
            self.save_model()



    def __is_saving_required(self):
        print("Not available now")




    def __save_model(self):

        # If we want to save to the pytorch format
        if self.method == "pytorch":
            try:
                torch.save(the_model.state_dict(), PATH)
            except:
                raise ValueError("Error while saving the pytorch model and weights")

        # If we want to save to the ONNX format
        elif self.method == "onnx":
            try:
                torch.onnx.export(self.model, dummy_input, "alexnet.proto", verbose=True, input_names=input_names, output_names=output_names)
            except:
                raise ValueError("Error while saving the ONNX model and weights")

        print("==> Model and weights saved")
