#
# Common imports
#
from typing import Union
from typing import List

#
# Backend imports
#
from torch.nn import Module
from torch import tensor
import torch

#
# Deeplodocus imports
#
from deeplodocus.utils.flags.entry import *
from deeplodocus.data.dataset import Dataset
from deeplodocus.core.inference.generic_inferer import GenericInferer
from deeplodocus.utils.flags.verbose import *
from deeplodocus.utils.generic_utils import get_corresponding_flag
from deeplodocus.utils.flags.flag_lists import DEEP_LIST_VERBOSE


class GenericEvaluator(GenericInferer):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    A GenericIEvaluator class
    """

    def __init__(self,
                 model: Module,
                 dataset: Dataset,
                 metrics: dict,
                 losses: dict,
                 batch_size: int = 4,
                 num_workers: int = 4,
                 verbose: Flag = DEEP_VERBOSE_BATCH):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize a GenericEvaluator instance

        PARAMETERS:
        -----------

        :param model->torch.nn.Module: The model to infer
        :param dataset->Dataset: A dataset
        :param batch_size->int: The number of instances per batch
        :param num_workers->int: The number of processes / threads used for data loading
        :param verbose->int: How verbose the class is


        RETURN:
        -------

        :return: None
        """

        #
        super().__init__(model=model,
                         dataset=dataset,
                         batch_size=batch_size,
                         num_workers=num_workers)
        self.verbose = get_corresponding_flag(DEEP_LIST_VERBOSE, verbose)
        self.verbose = verbose
        self.metrics = metrics
        self.losses = losses

    @staticmethod
    def compute_metrics(metrics: dict,
                        inputs: Union[tensor, List[tensor]],
                        outputs: Union[tensor, List[tensor]],
                        labels: Union[tensor, List[tensor]],
                        additional_data: Union[tensor, List[tensor]]) -> dict:
        """
        AUTHORS;
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Compute the metrics using the corresponding method arguments

        PARAMETERS:
        -----------

        :param metrics: dict: The metrics to compute
        :param inputs: Union[tensor, list]: The inputs
        :param outputs: Union[tensor, list]: Outputs of the network
        :param labels: Union[tensor, list]: Labels
        :param additional_data: Union[tensor, list]: Additional data

        RETURN:
        -------

        :return->dict: A dictionary containing the associations (key, output) of the metrics
        """

        result_metrics = {}

        # Temporary variable for saving the output
        temp_metric_result = None

        for key, metric in vars(metrics).items():
            metric_args = metric.get_arguments()
            metric_method = metric.get_method()

            # TODO : Check the number of arguments before compute the metric
            #if metric.is_loss() is True:
            #   num_required_args = len(inspect.getfullargspec(metric_method.forward)[0])
            #   num_given_args = len(metric_args + 1)
            #else:
            #   num_required_args = len(inspect.getfullargspec(metric_method)[0])
            #if num_required_args != num_given_args:
            #    Notification(DEEP_NOTIF_FATAL, "The metric %s takes %i positional arguments but %i were given" %(metric_method, len(num_required_args), len(metric_args)))

            #
            # Select the good type of input
            #
            if DEEP_ENTRY_INPUT in metric_args:
                if DEEP_ENTRY_LABEL in metric_args:
                    if DEEP_ENTRY_ADDITIONAL_DATA in metric_args:
                        temp_metric_result = metric_method(inputs, outputs, labels, additional_data)
                    else:
                        temp_metric_result = metric_method(inputs, outputs, labels)
                else:
                    if DEEP_ENTRY_ADDITIONAL_DATA in metric_args:
                        temp_metric_result = metric_method(inputs, outputs, additional_data)
                    else:
                        temp_metric_result = metric_method(inputs, outputs)
            else:
                if DEEP_ENTRY_LABEL in metric_args:
                    if DEEP_ENTRY_ADDITIONAL_DATA in metric_args:
                        temp_metric_result = metric_method(outputs, labels, additional_data)
                    else:
                        temp_metric_result = metric_method(outputs, labels)
                else:
                    if DEEP_ENTRY_ADDITIONAL_DATA in metric_args:
                        temp_metric_result = metric_method(outputs, additional_data)
                    else:
                        temp_metric_result = metric_method(outputs)

            #
            # Add the metric to the dictionary
            #

            # Check if the the metric is a Metric instance or a Loss instance
            if metric.is_loss() is True:
                # Do not call ".item()" in order to be able to achieve back propagation on the total_loss
                result_metrics[metric.get_name()] = temp_metric_result
            else:
                # If it is a Loss function we detach the tensor from the graph
                if isinstance(temp_metric_result, torch.nn.Module):
                    temp_metric_result = temp_metric_result.detach()
                result_metrics[metric.get_name()] = temp_metric_result.item()
        return result_metrics

