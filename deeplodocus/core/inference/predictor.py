import torch
from torch.nn import Module

from deeplodocus.data.load.dataset import Dataset
from deeplodocus.core.inference.generic_inferer import GenericInferer


class Predictor(GenericInferer):
    """
    AUTHORS:
    --------

    :author: Samuel Westlake
    :author: Alix Leroy


    DESCRIPTION:
    ------------

    A Predictor class which outputs the inferred result of the model
    """

    def __init__(self,
                 model: Module,
                 dataset: Dataset,
                 batch_size: int = 4,
                 num_workers: int = 4,
                 verbose: int = 2,
                 transform_manager=None):

        super().__init__(model=model,
                         dataset=dataset,
                         batch_size=batch_size,
                         num_workers=num_workers)
        self.transform_manager = transform_manager
        self.verbose = verbose

    def predict(self, model=None):
        """
        AUTHORS:
        --------

        :author: Samuel Westlake
        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Inference of the model

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return outputs->dict: the total losses and total metrics for the model over the test data set
        """
        self.model = self.model if model is None else model

        for minibatch_index, minibatch in enumerate(self.dataloader, 0):
            inputs, labels, additional_data = self.clean_single_element_list(minibatch)
            inputs, labels = self.to_device(inputs, model.device), self.to_device(labels, model.device)
            # Infer the outputs from the model over the given mini batch
            with torch.no_grad():
                outputs = self.model(*inputs)
            self.transform_manager.transform(outputs, inputs, labels, additional_data)
