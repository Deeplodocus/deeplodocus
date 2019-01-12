# Backend imports
import torch
import torch.nn as nn

# Deeplodocus imports
from deeplodocus.data.dataset import Dataset
from deeplodocus.core.inference.tester import Tester
from deeplodocus.utils.notification import Notification
from deeplodocus.utils.dict_utils import apply_weight
from deeplodocus.utils.dict_utils import sum_dict
from deeplodocus.core.inference.generic_evaluator import GenericEvaluator
from deeplodocus.brain.thalamus import Thalamus
from deeplodocus.brain.signal import Signal

# Deeplodocus flags
from deeplodocus.utils.flags import *
from deeplodocus.utils.flags.notif import *
from deeplodocus.utils.flags.event import *
from deeplodocus.utils.flags.shuffle import *


class Trainer(GenericEvaluator):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Trainer instance to train a model
    """
    def __init__(self,
                 model: nn.Module,
                 dataset: Dataset,
                 metrics: dict,
                 losses: dict,
                 optimizer,
                 num_epochs: int,
                 initial_epoch: int = 1,
                 batch_size: int = 4,
                 shuffle: Flag = DEEP_SHUFFLE_NONE,
                 num_workers: int = 4,
                 verbose: int = DEEP_VERBOSE_BATCH,
                 tester: Tester = None):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize a Trainer instance

        PARAMETERS:
        -----------

        :param model->torch.nn.Module: The model which has to be trained
        :param dataset->Dataset: The dataset to be trained on
        :param metrics->dict: The metrics to analyze
        :param losses->dict: The losses to use for the backpropagation
        :param optimizer: The optimizer to use for the backpropagation
        :param num_epochs->int: Number of epochs for the training
        :param initial_epoch->int: The index of the initial epoch
        :param batch_size->int: Size a minibatch
        :param shuffle->int: DEEP_SHUFFLE flag, method of shuffling to use
        :param num_workers->int: Number of processes / threads to use for data loading
        :param verbose->int: DEEP_VERBOSE flag, How verbose the Trainer is
        :param memorize->int: DEEP_MEMORIZE flag, what data to save
        :param save_condition->int: DEEP_SAVE flag, when to save the results
        :param tester->Tester: The tester to use for validation
        :param model_name->str: The name of the model

        RETURN:
        -------

        :return: None
        """
        # Initialize the GenericEvaluator par
        super().__init__(model=model,
                         dataset=dataset,
                         metrics=metrics,
                         losses=losses,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         verbose=verbose)
        self.shuffle = shuffle
        self.optimizer = optimizer
        self.initial_epoch = initial_epoch
        self.num_epochs = num_epochs

        if isinstance(tester, Tester):
            self.tester = tester          # Tester for validation
            self.tester.set_metrics(metrics=metrics)
            self.tester.set_losses(losses=losses)
        else:
            self.tester = None

        # Early stopping
        # self.stopping = Stopping(stopping_parameters)

        #
        # Connect signals
        #
        Thalamus().connect(receiver=self.saving_required,
                           event=DEEP_EVENT_SAVING_REQUIRED,
                           expected_arguments=["saving_required"])

    def fit(self, first_training: bool = True)->None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Fit the model to the dataset

        PARAMETERS:
        -----------

        :param first_training: (bool, optional): Whether it is the first training on the model or not

        RETURN:
        -------

        :return: None
        """
        self.__train(first_training=first_training)
        Notification(DEEP_NOTIF_SUCCESS, FINISHED_TRAINING)
        # Prompt if the user want to continue the training
        self.__continue_training()

    def __train(self, first_training=True) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Loop over the dataset to train the network

        PARAMETERS:
        -----------

        :param first_training->bool: Whether more epochs have been required after initial training or not

        RETURN:
        -------

        :return: None
        """
        if first_training is True:
            Thalamus().add_signal(signal=Signal(event=DEEP_EVENT_ON_TRAINING_START, args={}))
        else:
            self.callbacks.unpause()

        for epoch in range(self.initial_epoch+1, self.num_epochs+1):  # loop over the dataset multiple times

            Thalamus().add_signal(signal=Signal(event=DEEP_EVENT_ON_EPOCH_START, args={"epoch_index": epoch,
                                                                                       "num_epochs": self.num_epochs}))
            self.model.train()
            for minibatch_index, minibatch in enumerate(self.dataloader, 0):
                # Clean the given data
                inputs, labels, additional_data = self.clean_single_element_list(minibatch)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Infer the output of the batch
                # TODO: inputs should be a list and use outputs = self.model(*inputs) and put inputs on device
                inputs = inputs.to(device=self.model.device, dtype=torch.float)
                labels = labels.to(device=self.model.device, dtype=torch.long)

                outputs = self.model(inputs)

                # Compute losses and metrics
                result_losses = self.compute_metrics(self.losses, inputs, outputs, labels, additional_data)
                result_metrics = self.compute_metrics(self.metrics, inputs, outputs, labels, additional_data)

                # Add weights to losses
                result_losses = apply_weight(result_losses, self.losses)

                # Sum all the result of the losses
                total_loss = sum_dict(result_losses)

                # Accumulates the gradient (by addition) for each parameter
                total_loss.backward()

                # Performs a parameter update based on the current gradient (stored in .grad attribute of a parameter)
                # and the update rule
                self.optimizer.step()

                outputs, total_loss, result_losses, result_metrics = self.detach(outputs=outputs,
                                                                                 total_loss=total_loss,
                                                                                 result_losses=result_losses,
                                                                                 result_metrics=result_metrics)

                # Send signal batch end
                # Thalamus().add_signal(Signal(event= DEEP_EVENT_ON_BATCH_END,
                #                              args={"minibatch_index": minibatch_index+1,
                #                                    "num_minibatches": self.num_minibatches,
                #                                    "epoch_index": epoch,
                #                                    "total_loss": total_loss.item(),
                #                                    "result_losses": result_losses,
                #                                    "result_metrics": result_metrics
                #                                    }))

            # Shuffle the data if required
            if self.shuffle is not None:
                self.dataset.shuffle(self.shuffle)

            # Reset the dataset (transforms cache)
            self.dataset.reset()

            # Evaluate the model
            total_validation_loss, result_validation_losses, result_validation_metrics = self.__evaluate_epoch()

            # Send signal epoch end
            # Thalamus().add_signal(Signal(event=DEEP_EVENT_ON_EPOCH_END,
            #                              args={"epoch_index": epoch,
            #                                    "num_epochs": self.num_epochs,
            #                                    "model": self.model,
            #                                    "num_minibatches": self.num_minibatches,
            #                                    "total_validation_loss": total_validation_loss.item(),
            #                                    "result_validation_losses": result_validation_losses,
            #                                    "result_validation_metrics": result_validation_metrics,
            #                                    "num_minibatches_validation": self.tester.get_num_minibatches()
            #                                    }))
        # Send signal end training
        Thalamus().add_signal(Signal(event=DEEP_EVENT_ON_TRAINING_END,
                                     args={"model": self.model}))

    def detach(self, outputs, total_loss, result_losses, result_metrics):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Detach the tensors from the graph

        PARAMETERS:
        -----------

        :param outputs:
        :param total_loss:
        :param result_losses:
        :param result_metrics:

        RETURN:
        -------

        :return outputs:
        :return total_loss:
        :return result_losses:
        :return result_metrics:
        """

        total_loss = total_loss.detach()
        outputs = outputs.detach()

        for key, value in result_losses.items():
            result_losses[key] = value.detach()

        # Tensors already detached in compute metrics for more efficiency
        # for key, value in result_metrics.items():
        #     if isinstance(value, Tensor):
        #         result_metrics[key] = value.detach()

        return outputs, total_loss, result_losses, result_metrics

    def __continue_training(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Ask if we want to continue once the training ended

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        continue_training = ""
        # Ask if the user want to continue the training
        while continue_training.lower() not in ["y", "n"]:
            continue_training = Notification(DEEP_NOTIF_INPUT, "Would you like to continue the training ? (Y/N) ").get()
        # If yes ask the number of epochs
        if continue_training.lower() == "y":
            while True:
                epochs = Notification(DEEP_NOTIF_INPUT, "Number of epochs ? ").get()
                try:
                    epochs = int(epochs)
                    break
                except ValueError:
                    Notification(DEEP_NOTIF_WARNING, "Number of epochs must be an integer").get()
            if epochs > 0:
                self.initial_epoch = self.num_epochs + 1
                self.num_epochs += epochs
                # Resume the training
                self.fit(first_training=False)
        else:
            pass

    def __evaluate_epoch(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Evaluate the model using the tester

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: The total_loss, the individual losses and the individual metrics
        """
        # Initialize the losses and metrics results
        total_validation_loss = None
        result_losses = None
        result_metrics = None

        # If a tester is available compute the losses and metrics
        if self.tester is not None:
            total_validation_loss, result_losses, result_metrics = self.tester.evaluate(model=self.model)
        return total_validation_loss, result_losses, result_metrics

    def saving_required(self, saving_required: bool):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Signal to send the model to be saved if require

        PARAMETERS:
        -----------

        :param saving_required: (bool): Whether saving the model is required or not

        RETURN:
        -------
        """

        if saving_required is True:
            Thalamus().add_signal(signal=Signal(event=DEEP_EVENT_SAVE_MODEL, args={"model": self.model}))
