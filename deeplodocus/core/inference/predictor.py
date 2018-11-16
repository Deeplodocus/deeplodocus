from torch.utils.data import DataLoader
from torch.nn import Module
from torch import Tensor

from deeplodocus.data.dataset import Dataset


class Predictor(object):
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
                 verbose: int=2):


        self.model = model
        self.verbose = verbose
        self.dataloader = DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=num_workers)

    def predict(self):
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
        # Initialise an empty tensor to store outputs
        outputs = Tensor()
        for minibatch_index, minibatch in enumerate(self.dataloader, 0):
            inputs, labels, additional_data = self.__clean_single_element_list(minibatch)
            # Infer the outputs from the model over the given mini batch
            minibatch_output = self.model(*inputs)
            # Append mini_batch output to the output tensor
            outputs.cat(minibatch_output)
        return outputs

