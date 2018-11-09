from deeplodocus.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn import Module
from torch import Tensor


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
        # Initialise an empty tensor to store outputs
        outputs = Tensor()
        for minibatch_index, minibatch in enumerate(self.dataloader_test, 0):
            inputs, labels, additional_data = self.__clean_single_element_list(minibatch)
            # Infer the outputs from the model over the given mini batch
            minibatch_output = self.model(inputs)
            # Append mini_batch output to the output tensor
            outputs.cat(minibatch_output)
        return outputs

    @staticmethod
    def __clean_single_element_list(minibatch: list) -> list:
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

