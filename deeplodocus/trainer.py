from typing import Union
from typing import List

from torch import tensor
from torch.utils.data import  DataLoader

from deeplodocus.data.dataset import Dataset
from deeplodocus.callback import Callback
from deeplodocus.tester import Tester
from deeplodocus.utils.notification import Notification
from deeplodocus.utils.types import *
from deeplodocus.core.metric import Metric



class Trainer(object):

    def __init__(self, model,
                 dataset:Dataset,
                 metrics,
                 loss_functions:dict,
                 loss_weights:list,
                 optimizer,
                 epochs:int,
                 initial_epoch:int = 0,
                 batch_size:int = 4,
                 shuffle:bool = True,
                 num_workers = 4,
                 verbose=2,
                 save_condition="auto",
                 stopping_parameters=None,
                 write_logs=False):

        self.model = model
        self.write_logs=write_logs
        self.metrics = metrics
        self.loss_functions = loss_functions
        self.loss_weights = loss_weights
        self.optimizer = optimizer
        self.callbacks = Callback(model = model,
                                  optimizer=optimizer,
                                  metrics=metrics,
                                  initial_epoch=initial_epoch,
                                  working_directory="",
                                  model_name="test",
                                  verbose=verbose,
                                  save_condition=save_condition,
                                  stopping_parameters=stopping_parameters)
        self.epochs = epochs
        self.initial_epoch = initial_epoch
        self.train_dataset = dataset
        self.dataloader_train =  DataLoader(dataset=dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
        self.tester = Tester()          # Tester for validation


    def fit(self):
        """
        :param method:
        :return:
        """


        self.__train()

        Notification(DEEP_SUCCESS, "\n", write_logs=self.write_logs)
        Notification(DEEP_SUCCESS,"=============================================================", write_logs=self.write_logs)
        Notification(DEEP_SUCCESS,'Finished Training', write_logs=self.write_logs)
        Notification(DEEP_SUCCESS,"=============================================================", write_logs=self.write_logs)
        Notification(DEEP_SUCCESS,"\n", write_logs=self.write_logs)

        # Prompt if the user want to continue the training
        self.__continue_training()


    def __train(self, first_training = True):


        if first_training is True :

            self.callbacks.on_train_begin()


        for epoch in range(self.initial_epoch, self.epochs):  # loop over the dataset multiple times


            for i, batch in enumerate(self.dataloader_train, 0):

                # get the inputs
                inputs, labels, additional_data = self.__clean_single_element_list(batch)



                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model.forward(inputs)          #infer the outputs of the network


                result_losses = self.__compute_loss(self.loss_functions, outputs, labels, additional_data) # Compute the losses

                result_metrics = self.__compute_metrics(self.metrics, inputs, outputs, labels, additional_data)

                # Add weights to losses
                for i, loss in enumerate(result_losses):
                    result_losses[i] = loss * self.loss_weights[i]

                # Sum all the result of the losses
                total_loss = sum(result_losses)

                total_loss.backward() # accumulates the gradient (by addition) for each parameter
                self.optimizer.step() # performs a parameter update based on the current gradient (stored in .grad attribute of a parameter) and the update rule

                # Minibatch callback
                self.callbacks.on_batch_end(total_loss, result_losses, self.loss_weights, result_metrics)

            if self.shuffle is not None:
                pass

            # Reset the dataloader
            self.train_dataset.reset()

            #Epoch callback
            self.callbacks.on_epoch_end(um_total_epochs=num_total_epochs)

            self.__evaluate_val_dataset()

        # End of training callback
        self.callbacks.on_training_end()

    def __clean_single_element_list(self, batch:list)->list:
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
        cleaned_batch = []
        for entry in batch:
            if isinstance(entry, list) and len(entry) == 1:
                cleaned_batch.append(entry[0])
            elif isinstance(entry, list) and len(entry) ==0:
                cleaned_batch.append(None)
            else:
                cleaned_batch.append(entry)

        return  cleaned_batch


    def __compute_loss(self, criterions:dict, outputs:Union[tensor, list], labels:Union[tensor, list], additional_data:Union[tensor, list])->list:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Compute the different losses

        PARAMETERS:
        -----------

        :param criterions->dict: The different criterion f
        :param outputs: The predicted outputs
        :param labels:  The expected outputs
        :param additional_data: Additional data given to the loss function

        RETURN:
        -------

        :return losses->list: The list of computed losses
        """

        losses = []
        for _, criterion_function in criterions.items():        # First argument is criterion_name and is not needed here
            if labels is None:
                if additional_data is None:
                    losses.append(criterion_function(outputs))
                else:
                    losses.append(criterion_function(outputs, additional_data))
            else:
                if additional_data is None:
                    losses.append(criterion_function(outputs, labels))
                else:
                    losses.append(criterion_function(outputs, labels, additional_data))

        return losses


    def __compute_metrics(self, metrics:List[Metric], inputs, outputs, labels, additional_data)->list:
        result_metrics = []

        for metric in metrics:
            metric_args = metric.get_arguments()
            metric_method = metric.get_method()
            print(metric.get_method())

            if DEEP_ENTRY_INPUT in metric_args:
                if DEEP_ENTRY_LABEL in metric_args:
                    if DEEP_ENTRY_ADDITIONAL_DATA in metric_args:
                        result_metrics.append(metric_method(inputs, outputs, labels, additional_data))
                    else:
                        result_metrics.append(metric_method(inputs, outputs, labels))
                else:
                    if DEEP_ENTRY_ADDITIONAL_DATA in metric_args:
                        result_metrics.append(metric_method(inputs, outputs, additional_data))
                    else:
                        result_metrics.append(metric_method(inputs, outputs))
            else:
                if DEEP_ENTRY_LABEL in metric_args:
                    if DEEP_ENTRY_ADDITIONAL_DATA in metric_args:
                        result_metrics.append(metric_method(outputs, labels, additional_data))
                    else:
                        result_metrics.append(metric_method(outputs, labels))
                else:
                    if DEEP_ENTRY_ADDITIONAL_DATA in metric_args:
                        result_metrics.append(metric_method(outputs, additional_data))
                    else:
                        result_metrics.append(metric_method(outputs))

        return result_metrics

    def __continue_training(self):

        continue_training = ""

        # Ask if the user want to continue the training
        while continue_training.lower() != ("y" or "n"):

            continue_training = Notification(DEEP_INPUT, 'Would you like to continue the training ? (Y/N) ', write_logs=self.write_logs)

        #If yes ask the number of epochs
        if continue_training.lower() == "y":
            epochs = ""

            while not isinstance(epochs, int):
                epochs =  Notification(DEEP_INPUT, 'Number of epochs ? ', write_logs=self.write_logs)


        # Reset the system to continue the training
        if epochs > 0:
            self.initial_epoch = self.epochs
            self.epochs += epochs

            # Resume the training
            self.__train_from_file(first_training = False)





