import random

from deeplodocus.data.transformer.transformer import Transformer


class OneOf(Transformer):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    OneOf class inheriting from Transformer which compute one random transform from the list
    """
    def __init__(self, config):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize a OneOf transformer inheriting a Transformer

        PARAMETERS:
        -----------

        :param config->Namespace: The config

        RETURN:
        -------

        :return: None
        """
        Transformer.__init__(self, config)

    def transform(self, transformed_data, index):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Transform the data using the One Of transformer

        PARAMETERS:
        -----------

        :param data: The data to transform
        :param index: The index of the data

        RETURN:
        -------

        :return transformed_data: The transformed data
        """
        transforms = []
        if self.last_index == index:
            transforms = self.last_transforms

        else:
            transforms += self.list_mandatory_transforms                                    # Get the mandatory transforms
            random_transform_index = random.randint(0, len(self.list_transforms) -1)        # Get a random transform among the ones available in the list
            transforms += self.list_transforms[random_transform_index]                      # Get the one function

        # Reinitialize the last transforms
        self.last_transforms = []

        # Apply the transforms
        transformed_data = self.apply_transforms(transformed_data, transforms)

        self.last_index = index
        return transformed_data
