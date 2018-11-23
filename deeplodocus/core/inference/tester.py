#
# COMMON IMPORTS
#



#
# BACKEND IMPORTS
#

from torch.nn import Module


#
# DEEPLDODOCUS IMPORTS
#

from deeplodocus.data.dataset import Dataset
from deeplodocus.utils import dict_utils
from deeplodocus.utils.flags import *
from deeplodocus.core.inference.generic_evaluator import GenericEvaluator


class Tester(GenericEvaluator):
    """
    AUTHORS:
    --------

    :author: Samuel Westlake
    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Class for evaluating the results of the model
    """

    def __init__(self,
                 model: Module,
                 dataset: Dataset,
                 metrics: dict,
                 losses: dict,
                 batch_size: int = 4,
                 num_workers: int = 4,
                 verbose: int = DEEP_VERBOSE_BATCH):

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
        :param losses->dict: The losses to use for the backpropagation
        :param batch_size->int: Size a minibatch
        :param num_workers->int: Number of processes / threads to use for data loading
        :param verbose->int: DEEP_VERBOSE flag, How verbose the Trainer is


        RETURN:
        -------

        :return: None
        """

        super().__init__(model=model,
                         dataset=dataset,
                         metrics=metrics,
                         losses=losses,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         verbose=verbose)


    def evaluate(self):
        """
        AUTHORS:
        --------

        :author: Samuel Westlake
        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize a Tester instance

        PARAMETERS:
        -----------

        :param model->torch.nn.Module: The model which has to be trained
        :param dataset->Dataset: The dataset to be trained on
        :param metrics->dict: The metrics to analyze
        :param losses->dict: The losses to use for the backpropagation
        :param batch_size->int: Size a minibatch
        :param num_workers->int: Number of processes / threads to use for data loading
        :param verbose->int: DEEP_VERBOSE flag, How verbose the Trainer is


        RETURN:
        -------

        :return sum_losses->dict: Sum of the losses over the test data set
        :return total_losses->dict: Total losses for the model over the test data set
        :return total_metrics->dict: Total metrics for the model over the test data set
        """

        # Make dictionaries like losses and metrics but initialised with lists
        total_losses = dict_utils.like(self.losses, [])
        total_metrics = dict_utils.like(self.metrics, [])

        # Loop through each mini batch
        for minibatch_index, minibatch in enumerate(self.dataloader, 0):

            # Get the data
            inputs, labels, additional_data = self.clean_single_element_list(minibatch)

            # Infer the outputs from the model over the given mini batch
            outputs = self.model(*inputs)

            # Compute the losses and metrics
            batch_losses = self.compute_metrics(self.losses, inputs, outputs, labels, additional_data)
            batch_metrics = self.compute_metrics(self.metrics, inputs, outputs, labels, additional_data)

            # Apply weights to the losses
            batch_losses = dict_utils.apply_weight(batch_losses, self.losses)

            # Append the losses and metrics for this batch to the total losses and metrics
            total_losses = dict_utils.apply(total_losses, batch_losses, "append")
            total_metrics = dict_utils.apply(total_metrics, batch_metrics, "append")

        # Calculate the mean for each loss and metric
        total_losses = dict_utils.mean(total_losses)
        total_metrics = dict_utils.mean(total_metrics)

        # Calculate the sum of the losses
        sum_losses = dict_utils.sum_dict(total_losses)

        return sum_losses, total_losses, total_metrics


    def set_metrics(self, metrics:dict):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Setter for self.metrics

        PARAMETERS:
        -----------

        :param metrics->dict: The metrics we want to analyze

        RETURN:
        -------

        :return: None
        """
        self.metrics = metrics

    def set_losses(self, losses:dict):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Setter for self.losses

        PARAMETERS:
        -----------

        :param losses->dict: The losses we want to analyze

        RETURN:
        -------

        :return: None
        """
        self.losses = losses
