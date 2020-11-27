from typing import Tuple
from typing import Union
from typing import List

from torch.utils.data import DataLoader
from torch.nn import Module
from torch import Tensor

from deeplodocus.data.load.dataset import Dataset
from deeplodocus.utils.namespace import Namespace


class GenericInferer(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    A GenericInferer class
    """

    def __init__(self,
                 model: Module,
                 dataset: Dataset,
                 batch_size: int = 4,
                 num_workers: int = 4):

        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize a GenericInferer instance

        PARAMETERS:
        -----------

        :param model (torch.nn.Module): The model to infer
        :param dataset (Dataset): A dataset
        :param batch_size (int): The number of instances per batch
        :param num_workers (int): The number of processes / threads used for data loading
        """

        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        self.num_minibatches = self.compute_num_minibatches(
            batch_size=batch_size,
            length_dataset=dataset.__len__()
        )

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

    @staticmethod
    def compute_num_minibatches(length_dataset: int, batch_size: int) -> int:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Calculate the number of batches for each epoch

        PARAMETERS:
        -----------

        :param length_dataset(int): The length of the dataset
        :param batch_size(int): The number of instances in one mini-batch

        RETURN:
        -------

        :return: None
        """
        if length_dataset % batch_size == 0:
            num_minibatches = length_dataset // batch_size
        else:
            num_minibatches = (length_dataset // batch_size) + 1

        return num_minibatches

    def get_num_minibatches(self) -> int:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Getter for self.num_minibatches

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return self.num_minibatches->int: The number of mini batches in the Inferer instance
        """
        return self.num_minibatches

    def to_device(self, data, device):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Convert data to the correct backend format

        PARAMETERS:
        -----------

        :param data (data):
        :param device:

        RETURN:
        -------

        :return:
        """
        if isinstance(data, list):
            l_data = []
            for d in data:
                td = self.to_device(d, device)
                l_data.append(td)
            return l_data

        else:
            try:
                return data.to(device)
            except AttributeError:
                try:
                    return [d.to(device=device) for d in data if d is not None]
                except TypeError:
                    return None

    def recursive_detach(self, outputs: Union[Tuple, List, Tensor]) ->Union[Tuple, List, Tensor]:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Recusively detach tensors whether they are located in lists or tuples

        PARAMETERS:
        -----------


        :param outputs:


        :return:
        """
        if isinstance(outputs, list):
            for i, o in enumerate(outputs):
                outputs[i] = self.recursive_detach(o)
        elif isinstance(outputs, tuple):
            tuple_list = []
            for o in outputs:
                i = self.recursive_detach(o)
                tuple_list.append(i)
            outputs = tuple(tuple_list)
        elif isinstance(outputs, dict):
            detached_outputs = {}
            for key, item in outputs.items():
                try:
                    detached_outputs[key] = self.recursive_detach(item)
                except AttributeError:
                    detached_outputs[key] = item
            outputs = detached_outputs
        elif isinstance(outputs, Namespace):
            detached_outputs = {}
            for key, item in outputs.__dict__.items():
                try:
                    detached_outputs[key] = self.recursive_detach(item)
                except AttributeError:
                    detached_outputs[key] = item
            outputs = Namespace(detached_outputs)
        else:
            outputs = outputs.detach()
        return outputs
