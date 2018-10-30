from torch.utils.data import  DataLoader

from deeplodocus.data.dataset import Dataset
from deeplodocus.callback import Callback
from deeplodocus.tester import Tester
from deeplodocus.utils.notification import Notification
from deeplodocus.utils.types import *



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
                inputs, labels, additional_data = batch

                # zero the parameter gradients
                self.optimizer.zero_grad()
                print(inputs.shape)
                # forward + backward + optimize
                outputs = self.model(inputs)          #infer the outputs of the network

                criterions = self.loss_functions

                losses = self.__compute_loss(criterions, outputs, labels, additional_data) # Compute the losses
                metrics = self.__compute_metrics(outputs, labels)

                # Add weights to losses
                for i, loss in enumerate(self.losses):
                    loss *= self.losses_weights[i]

                # Sum all the losses
                loss = sum(self.losses)

                loss.backward() # accumulates the gradient (by addition) for each parameter
                self.optimizer.step() # performs a parameter update based on the current gradient (stored in .grad attribute of a parameter) and the update rule

                # Minibatch callback
                self.callbacks.on_batch_end(losses, metrics)

            if self.shuffle is not None:
                pass

            # Reset the dataloader
            self.train_dataset.reset()
            #Epoch callback
            self.callbacks.on_epoch_end(um_total_epochs=num_total_epochs)

            self.__evaluate_val_dataset()

        # End of training callback
        self.callbacks.on_training_end()



    def __compute_loss(self, criterion, outputs, labels, additional_data):

        if labels is None:
            if additional_data is None:
                loss = criterion(outputs)
            else:
                loss = criterion(outputs, additional_data)
        else:
            if additional_data is None:
                loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs, labels, additional_data)

        return loss

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





