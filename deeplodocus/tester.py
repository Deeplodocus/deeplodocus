from deeplodocus.data.dataset import Dataset
from deeplodocus.utils import dict_utils
from torch.utils.data import DataLoader
from torch.nn import Module


class Tester(object):
    """
    Author: Samuel Westlake, Alix Leroy
    """

    def __init__(self,
                 model: Module,
                 dataset: Dataset,
                 metrics: dict,
                 losses: dict,
                 batch_size: int = 4,
                 num_workers: int = 4,
                 verbose: int=2):
        self.model = model
        self.metrics = metrics
        self.losses = losses
        self.test_dataset = dataset
        self.verbose = verbose
        self.dataloader_test = DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=num_workers)

    def evaluate(self):
        """
        Author: SW
        :return: dict, dict: the total losses and total metrics for the model over the test data set
        """
        # Make dictionaries like losses and metrics but initialised with lists
        total_losses = dict_utils.like(self.losses, [])
        total_metrics = dict_utils.like(self.metrics, [])
        # Loop through each mini batch
        for minibatch_index, minibatch in enumerate(self.dataloader_test, 0):
            inputs, labels, additional_data = self.__clean_single_element_list(minibatch)
            # Infer the outputs from the model over the given mini batch
            outputs = self.model(inputs)
            # Compute the losses and metrics
            batch_losses = self.__compute_metrics(self.losses, inputs, outputs, labels, additional_data)
            batch_metrics = self.__compute_metrics(self.metrics, inputs, outputs, labels, additional_data)
            # Apply weights to the losses
            batch_losses = dict_utils.apply_weight(batch_losses, self.losses)
            # Append the losses and metrics for this batch to the total losses and metrics
            total_losses = dict_utils.apply(total_losses, batch_losses, "append")
            total_metrics = dict_utils.apply(total_metrics, batch_metrics, "append")
        # Calculate the mean for each loss and metric
        total_losses = dict_utils.mean(total_losses)
        total_metrics = dict_utils.mean(total_metrics)
        return dict_utils.sum_dict(total_losses), total_losses, total_metrics

    def __clean_single_element_list(self, minibatch:list)->list:
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

    @staticmethod
    def __compute_metrics(metrics: dict,
                          inputs: Union[tensor, list],
                          outputs: Union[tensor, list],
                          labels: Union[tensor, list],
                          additional_data: Union[tensor, list]) -> dict:
        """
        AUTHORS;
        --------
        :author: Alix Leroy
        DESCRIPTION:
        ------------
        Compute the metrics using the corresponding method arguments
        PARAMETERS:
        -----------
        :param metrics->dict: The metrics to compute
        :param inputs->Union[tensor, list]: The inputs
        :param outputs->Union[tensor, list]: Outputs of the network
        :param labels->Union[tensor, list]: Labels
        :param additional_data->Union[tensor, list]: Additional data
        RETURN:
        -------
        :return->dict: A dictionary containing the associations (key, output)
        """

        result_metrics = {}

        # Temporary variable for saving the output
        temp_metric_result = None

        for metric in list(metrics.values()):
            metric_args = metric.get_arguments()
            metric_method = metric.get_method()

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
                result_metrics[metric.get_name()] = temp_metric_result.item()

        return result_metrics
