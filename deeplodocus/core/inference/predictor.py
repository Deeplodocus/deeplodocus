from torch.nn import Module
from torch import Tensor

from deeplodocus.data.dataset import Dataset
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
                 verbose: int = 2):

        super().__init__(model=model,
                         dataset=dataset,
                         batch_size=batch_size,
                         num_workers=num_workers)

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

        # Initialise an empty list to store outputs
        inputs = []
        outputs = []
        for minibatch_index, minibatch in enumerate(self.dataloader, 0):
            inp, labels, additional_data = self.clean_single_element_list(minibatch)
            inp, labels = self.to_device(inp, model.device), self.to_device(labels, model.device)
            # Infer the outputs from the model over the given mini batch
            minibatch_output = self.model(*inp)
            # Append mini_batch output to the output tensor
            outputs.append(self.recursive_detach(minibatch_output))
            inputs.append(inp)
        return inputs, outputs

