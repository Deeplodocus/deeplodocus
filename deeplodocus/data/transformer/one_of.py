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

    def transform(self, data, index, data_type):
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
        :param data_type: The data_type

        RETURN:
        -------

        :return transformed_data: The transformed data
        """
        if self.last_index == index:
            transform = self.last_transforms[0]

        else:
            random_transform_index = random.randint(0, len(self.list_transforms) -1)        # Get a random transform among the ones available in the list
            transform= self.list_transforms[random_transform_index]                         # Get the function

        transform_name = transform[0]
        transform_method = transform[1]             # Create a generic alias for the transform method
        transform_args = transform[2]                                   # Dictionary of arguments
        transform_data, last_method_used = transform_method(data, **transform_args)

        # Reinitialize the last transforms
        self.last_transforms = []

        # Update the last transforms used and the last index
        if last_method_used is None:
            self.last_transforms.append([transform_name, transform_method, transform_args])

        else:
            self.last_transforms.append(last_method_used)

        self.last_index = index
        return transform_data
