from deeplodocus.data.dataset import Dataset
from deeplodocus.utils.dict_utils import apply, apply_weight
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
        :return:
        """
        total_losses = {}
        for name in list(self.losses.keys()):
            total_losses[name] = []
        total_metrics = {}
        for name in list(self.metrics.keys()):
            total_metrics[name] = []
        for minibatch_index, minibatch in enumerate(self.dataloader_test, 0):
            inputs, labels, additional_data = self.__clean_single_element_list(minibatch)
            outputs = self.model(inputs)
            batch_losses = self.__compute_losses(self.losses, inputs, outputs, labels, additional_data)
            batch_metrics = self.__compute_metrics(self.metrics, inputs, outputs, labels, additional_data)
            batch_losses = apply_weight(batch_losses, self.losses)
            total_losses = apply(total_losses, batch_losses, "append")
            total_metrics = apply(total_metrics, batch_metrics, "append")
        total_losses =
        return total_losses, total_metrics


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


    def __compute_losses(self, losses:dict, inputs:Union[tensor, list], outputs:Union[tensor, list], labels:Union[tensor, list], additional_data:Union[tensor, list])->dict:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Compute the different losses

        PARAMETERS:
        -----------

        :param loss_functions->List[Loss]: The different loss functions
        :param outputs: The predicted outputs
        :param labels:  The expected outputs
        :param additional_data: Additional data given to the loss function

        RETURN:
        -------

        :return losses->dict: The list of computed losses
        """

        result_losses = {}
        temp_loss_result = None

        for loss in list(losses.values()):
            loss_args = loss.get_arguments()
            loss_method = loss.get_method()

            if DEEP_ENTRY_INPUT in loss_args:
                if DEEP_ENTRY_LABEL in loss_args:
                    if DEEP_ENTRY_ADDITIONAL_DATA in loss_args:
                        temp_loss_result = loss_method(inputs, outputs, labels, additional_data)
                    else:
                        temp_loss_result = loss_method(inputs, outputs, labels)
                else:
                    if DEEP_ENTRY_ADDITIONAL_DATA in loss_args:
                        temp_loss_result = loss_method(inputs, outputs, additional_data)
                    else:
                        temp_loss_result = loss_method(inputs, outputs)
            else:
                if DEEP_ENTRY_LABEL in loss_args:
                    if DEEP_ENTRY_ADDITIONAL_DATA in loss_args:
                        temp_loss_result = loss_method(outputs, labels, additional_data)
                    else:
                        temp_loss_result = loss_method(outputs, labels)
                else:
                    if DEEP_ENTRY_ADDITIONAL_DATA in loss_args:
                        temp_loss_result = loss_method(outputs, additional_data)
                    else:
                        temp_loss_result = loss_method(outputs)

            # Add the loss to the dictionary
            result_losses[loss.get_name()] = temp_loss_result
        return result_losses


    def __compute_metrics(self, metrics:dict, inputs:Union[tensor, list], outputs:Union[tensor, list], labels:Union[tensor, list], additional_data:Union[tensor, list])->dict:


        result_metrics = {}
        temp_metric_result = None

        for metric in list(metrics.values()):
            metric_args = metric.get_arguments()
            metric_method = metric.get_method()

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

            # Add the metric to the dictionary
            result_metrics[metric.get_name()] = temp_metric_result.item()
        return result_metrics