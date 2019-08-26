# Python imports
import weakref

# Backend imports
import torch.nn as nn

# Deeplodocus imports
from deeplodocus.data.load.dataset import Dataset
from deeplodocus.core.inference.tester import Tester
from deeplodocus.utils.notification import Notification
import deeplodocus.utils.dict_utils as dict_utils
from deeplodocus.utils.dict_utils import sum_dict
from deeplodocus.core.inference.generic_evaluator import GenericEvaluator
from deeplodocus.brain.thalamus import Thalamus
from deeplodocus.brain.signal import Signal

# Deeplodocus flags
from deeplodocus.flags import *
from deeplodocus.utils.generic_utils import get_corresponding_flag


class Trainer(GenericEvaluator):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Trainer instance to train a model

    PUBLIC METHOD:
    --------------
    :method fit: Start the training
    :method detach: Detach the output tensors from the model in order to avoid memory leaks
    :method continue_training: Continue the training of the model
    :method saving_required: Send a signal to the Saver in order to save the model
    :method send_save_params: Send the parameters to the Saver

    PRIVATE METHOD:
    ---------------
    :method __init__: Initialize the Trainer
    :method __train: Loop over the dataset to train the network
    :method  __evaluate_epoch: Evaluate the model using the Validator

    """

    """
    "
    " PRIVATE METHODS
    "
    """

    def __init__(
            self,
            model: nn.Module,
            dataset: Dataset,
            metrics: dict,
            losses: dict,
            optimizer,
            num_epochs: int,
            initial_epoch: int = 1,
            batch_size: int = 4,
            shuffle_method: Flag = DEEP_SHUFFLE_NONE,
            num_workers: int = 4,
            verbose: Flag = DEEP_VERBOSE_BATCH,
            tester: Tester = None,
            transform_manager=None
    ) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize a Trainer instance

        PARAMETERS:
        -----------

        :param model (torch.nn.Module): The model which has to be trained
        :param dataset (Dataset): The dataset to be trained on
        :param metrics (dict): The metrics to analyze
        :param losses (dict): The losses to use for the backpropagation
        :param optimizer: The optimizer to use for the backpropagation
        :param num_epochs (int): Number of epochs for the training
        :param initial_epoch (int): The index of the initial epoch
        :param batch_size (int): Size a minibatch
        :param shuffle_method (Flag): DEEP_SHUFFLE flag, method of shuffling to use
        :param num_workers (int): Number of processes / threads to use for data loading
        :param verbose (int): DEEP_VERBOSE flag, How verbose the Trainer is
        :param memorize (int): DEEP_MEMORIZE flag, what data to save
        :param save_condition (int): DEEP_SAVE flag, when to save the results
        :param tester (Tester): The tester to use for validation
        :param model_name (str): The name of the model

        RETURN:
        -------

        :return: None
        """
        # Initialize the GenericEvaluator par
        super().__init__(
            model=model,
            dataset=dataset,
            metrics=metrics,
            losses=losses,
            batch_size=batch_size,
            num_workers=num_workers,
            verbose=verbose
        )
        self.optimizer = optimizer
        self.initial_epoch = initial_epoch
        self.epoch = None
        self.validation_loss = None
        self.num_epochs = num_epochs
        self.transform_manager = transform_manager

        # Load shuffling method
        self.shuffle_method = get_corresponding_flag(
            DEEP_LIST_SHUFFLE,
            shuffle_method,
            fatal=False,
            default=DEEP_SHUFFLE_NONE
        )

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
        Thalamus().connect(
            receiver=self.saving_required,
            event=DEEP_EVENT_SAVING_REQUIRED,
            expected_arguments=["saving_required"]
        )
        Thalamus().connect(
            receiver=self.send_save_params,
            event=DEEP_EVENT_REQUEST_SAVE_PARAMS_FROM_TRAINER,
            expected_arguments=[]
        )

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
            return total_validation_loss.item(), result_losses, result_metrics
        else:
            return total_validation_loss, result_losses, result_metrics

    def __train(self, first_training: bool = True) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Loop over the dataset to train the network

        PARAMETERS:
        -----------

        :param first_training (bool): Whether more epochs have been required after initial training or not

        RETURN:
        -------

        :return: None
        """
        if first_training is True:
            Thalamus().add_signal(signal=Signal(event=DEEP_EVENT_ON_TRAINING_START, args={}))

        for self.epoch in range(self.initial_epoch + 1, self.num_epochs + 1):

            Thalamus().add_signal(
                signal=Signal(
                    event=DEEP_EVENT_ON_EPOCH_START,
                    args={
                        "epoch_index": self.epoch,
                        "num_epochs": self.num_epochs
                    }
                )
            )

            # Shuffle the data if required
            if self.shuffle_method is not None:
                self.dataset.shuffle(self.shuffle_method)

            # Put model into train mode for the start of the epoch
            self.model.train()

            for minibatch_index, minibatch in enumerate(self.dataloader, 0):

                # Clean the given data
                inputs, labels, additional_data = self.clean_single_element_list(minibatch)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Set the data to the corresponding device
                inputs = self.to_device(inputs, self.model.device)
                labels = self.to_device(labels, self.model.device)
                additional_data = self.to_device(additional_data, self.model.device)

                # Infer the output of the batch
                try:
                    outputs = self.model(*inputs)
                except RuntimeError as e:
                    Notification(DEEP_NOTIF_FATAL, "RuntimeError : %s" % str(e))
                except TypeError as e:
                    Notification(DEEP_NOTIF_FATAL, "TypeError : %s" % str(e))

                # Compute losses
                result_losses = self.compute_metrics(self.losses, inputs, outputs, labels, additional_data)

                # Transform outputs
                if self.transform_manager is not None:
                    outputs = self.transform_manager.transform(outputs)

                # Compute metrics
                result_metrics = self.compute_metrics(self.metrics, inputs, outputs, labels, additional_data)

                # Add weights to losses
                result_losses = dict_utils.apply_weight(result_losses, vars(self.losses))

                # Sum all the result of the losses
                total_loss = sum_dict(result_losses)

                # Accumulates the gradient (by addition) for each parameter
                total_loss.backward()

                # Performs a parameter update based on the current gradient (stored in .grad attribute of a parameter)
                # and the update rule
                self.optimizer.step()

                # Detach the tensors from the network
                outputs, total_loss, result_losses, result_metrics = self.detach(
                    outputs=outputs,
                    total_loss=total_loss,
                    result_losses=result_losses,
                    result_metrics=result_metrics
                )

                # Send signal batch end
                Thalamus().add_signal(
                    Signal(
                        event=DEEP_EVENT_ON_BATCH_END,
                        args={
                            "minibatch_index": minibatch_index + 1,
                            "num_minibatches": self.num_minibatches,
                            "epoch_index": self.epoch,
                            "total_loss": total_loss.item(),
                            "result_losses": result_losses,
                            "result_metrics": result_metrics
                        }
                    )
                )

            # Reset the dataset (transforms cache)
            self.dataset.reset()

            # Evaluate the model
            self.validation_loss, result_validation_losses, result_validation_metrics = self.__evaluate_epoch()

            if self.tester is not None:
                num_minibatches_validation = self.tester.get_num_minibatches()
            else:
                num_minibatches_validation = None

            # Send signal epoch end
            Thalamus().add_signal(
                Signal(
                    event=DEEP_EVENT_ON_EPOCH_END,
                    args={
                        "epoch_index": self.epoch,
                        "num_epochs": self.num_epochs,
                        "model": weakref.ref(self.model),
                        "num_minibatches": self.num_minibatches,
                        "total_validation_loss": self.validation_loss,
                        "result_validation_losses": result_validation_losses,
                        "result_validation_metrics": result_validation_metrics,
                        "num_minibatches_validation": num_minibatches_validation,
                    }
                )
            )

        # Send signal end training
        Thalamus().add_signal(
            Signal(
                event=DEEP_EVENT_ON_TRAINING_END,
                args={"model": self.model}
            )
        )

    """
    "
    " PUBLIC METHODS
    "
    """

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
        Notification(DEEP_NOTIF_INFO, DEEP_MSG_TRAINING_STARTED)
        self.__train(first_training=first_training)
        Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_TRAINING_FINISHED)

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

        # Detach the total loss
        total_loss = total_loss.detach()

        # Detach the outputs recursively (Tuple + list)
        outputs = self.recursive_detach(outputs)

        # Detach the losses
        for key, value in result_losses.items():
            result_losses[key] = value.detach()

        # Metric tensors already detached in self.compute_metrics for more efficiency
        # Please keep these commented line in order not to forget
        # for key, value in result_metrics.items():
        #     if isinstance(value, Tensor):
        #         result_metrics[key] = value.detach()

        return outputs, total_loss, result_losses, result_metrics

    def continue_training(self, epochs=None):
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Function to know the number of epochs when continuing the training

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        if epochs is None:
            # If the user wants to continue the training, ask the number of epochs
            while True:
                epochs = Notification(DEEP_NOTIF_INPUT, "Number of epochs ? ").get()
                try:
                    epochs = int(epochs)
                    break
                except ValueError:
                    Notification(DEEP_NOTIF_WARNING, "Number of epochs must be an integer").get()
        if epochs > 0:
            self.initial_epoch = self.num_epochs
            self.num_epochs += epochs
            # Resume the training
            self.fit(first_training=False)

    def saving_required(self, saving_required: bool):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Signal to send the model to be saved if require
        NB : Contains a signal, cannot be static

        PARAMETERS:
        -----------

        :param saving_required: (bool): Whether saving the model is required or not

        RETURN:
        -------

        None
        """
        if saving_required is True:
            Thalamus().add_signal(
                signal=Signal(
                    event=DEEP_EVENT_SAVE_MODEL,
                    args={}
                )
            )

    def send_save_params(self, inp=None) -> None:
        """
        AUTHORS:
        --------

        :author: Samuel Westlake
        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Send the saving parameters to the Saver

        PARAMETERS:
        -----------

        :param inp: The input size of the model (required for ONNX models)

        RETURN:
        -------

        :return: None
        """
        Thalamus().add_signal(
            Signal(
                event=DEEP_EVENT_SEND_SAVE_PARAMS_FROM_TRAINER,
                args={"model": self.model,
                      "optimizer": self.optimizer,
                      "epoch_index": self.epoch,
                      "validation_loss": self.validation_loss,
                      "inp": inp}
            )
        )

