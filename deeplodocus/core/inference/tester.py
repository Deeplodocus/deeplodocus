import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from deeplodocus.data.load.dataset import Dataset
from deeplodocus.flags import *
from deeplodocus.utils.namespace import Namespace


class Tester(object):
    """
    AUTHORS:
    --------

    :author: Samuel Westlake
    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Class for evaluating the results of the model
    """

    def __init__(
            self,
            name: str,
            model: nn.Module,
            dataset: Dataset,
            metrics: dict,
            losses: dict,
            batch_size: int = 4,
            num_workers: int = 4,
            verbose: int = DEEP_VERBOSE_BATCH,
            transform_manager=None
    ):

        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize a Tester instance

        PARAMETERS:
        -----------

        :param model->torch.nn.Module: The model which has to be trained
        :param dataset->Dataset: The dataset to be trained on
        :param metrics->dict: The metrics to analyze
        :param losses->dict: The losses to use for the back-propagation
        :param batch_size->int: Size a mini-batch
        :param num_workers->int: Number of processes / threads to use for data loading
        :param verbose->int: DEEP_VERBOSE flag, How verbose the Trainer is


        RETURN:
        -------

        :return: None
        """
        self.name = name
        self.model = model
        self.dataset = dataset
        self.metrics = metrics
        self.losses = losses
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.verbose = verbose
        self.transform_manager = transform_manager
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

    def evaluate(self):
        """
        AUTHORS:
        --------

        :author: Samuel Westlake
        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Evaluate the model on the full dataset

        PARAMETERS:
        -----------

        :param model: (torch.nn.Module): The model which has to be trained

        RETURN:
        -------

        :return sum_losses (dict): Sum of the losses over the test data set
        :return total_losses (dict): Total losses for the model over the test data set
        :return total_metrics (dict): Total metrics for the model over the test data set
        """

        self.model.eval()  # Put model into evaluation mode
        self.losses.reset(self.dataset.type)  # Reset corresponding losses
        self.metrics.reset(self.dataset.type)  # Reset corresponding metrics

        # Loop through each mini batch
        for minibatch_index, minibatch in enumerate(self.dataloader, 0):
            inputs, labels, additional_data = self.clean_single_element_list(minibatch)

            # Set the data to the corresponding device
            inputs = self.to_device(inputs, self.model.device)
            labels = self.to_device(labels, self.model.device)
            additional_data = self.to_device(labels, self.model.device)

            # Infer the outputs from the model over the given mini batch
            with torch.no_grad():
                outputs = self.model(*inputs)
            outputs = self.detach(outputs)  # Detach the tensor from the graph

            # Compute the losses
            self.losses.forward(self.dataset.type, outputs, labels, inputs, additional_data)

            # Compute output transforms
            if self.transform_manager is not None:
                outputs = self.transform_manager.transform(
                    inputs=inputs,
                    outputs=outputs,
                    labels=labels,
                    additional_data=additional_data
                )

            # Compute the metrics
            self.metrics.forward(self.dataset.type, outputs, labels, inputs, additional_data)

        self.transform_manager.finish()  # Call finish on all output transforms
        loss, losses = self.losses.reduce(self.dataset.type)  # Get total loss and mean of each loss
        metrics = self.metrics.reduce(self.dataset.type)  # Get total metric values

        return loss.item(), losses, metrics

    @staticmethod
    def clean_single_element_list(minibatch: list) -> list:
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
        cleaned_minibatch = []
        # For each entry in the minibatch:
        # If it is a single element list -> Make it the single element
        # If it is an empty list -> Make it None
        # Else -> Do not change
        for entry in minibatch:
            if isinstance(entry, list) and len(entry) == 1:
                cleaned_minibatch.append(entry[0])
            elif isinstance(entry, list) and len(entry) == 0:
                cleaned_minibatch.append(None)
            else:
                cleaned_minibatch.append(entry)
        return cleaned_minibatch

    def to_device(self, x, device):
        if isinstance(x, list):
            x_ = []
            for item in x:
                item = self.to_device(item, device)
                x_.append(item)
            return x_

        else:
            try:
                return x.to(device)
            except AttributeError:
                try:
                    return [item.to(device=device) for item in x if item is not None]
                except TypeError:
                    return None

    def detach(self, x):
        if isinstance(x, list):
            x = [self.detach(item) for item in x]
        elif isinstance(x, tuple):
            x = tuple([self.detach(item) for item in x])
        elif isinstance(x, dict):
            x = {key: self.detach(item) for key, item in x.items()}
        elif isinstance(x, Namespace):
            x = {key: self.detach(item) for key, item in x.__dict__.items()}
        else:
            try:
                x = x.detach()
            except AttributeError:
                pass
        return x

