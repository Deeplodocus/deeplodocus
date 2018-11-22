from .transformer import Transformer

class Sequential(Transformer):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Sequential class inheriting from Transformer which compute the list of transforms sequentially
    """

    def __init__(self, config):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize a Sequential transformer inheriting a Transformer

        PARAMETERS:
        -----------

        :param config->Namespace: The config

        RETURN:
        -------

        :return: None
        """
        Transformer.__init__(self, config)


    def transform(self, transformed_data, index, data_type):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Transform the data using the Sequential transformer

        PARAMETERS:
        -----------

        :param transformed_data: The data to transform
        :param index: The index of the data
        :param data_type: The data_type

        RETURN:
        -------

        :return transformed_data: The transformed data
        """
        transforms = []

        if self.last_index == index:
            transforms = self.last_transforms

        else:
            transforms = self.list_transforms

        # Reinitialize the last transforms
        self.last_transforms = []


        # Apply the transforms
        for transform in transforms:

            transform_name = transform[0]
            transform_method = transform[1]  # Create a generic alias for the transform method
            transform_args = transform[2]  # Dictionary of arguments
            transformed_data, last_method_used = transform_method(transformed_data, **transform_args)       # Apply the transform

            # Update the last transforms used and the last index
            if last_method_used is None:
                self.last_transforms.append([transform_name, transform_method, transform_args])

            else:
                self.last_transforms.append(last_method_used)

        # Update the last index
        self.last_index = index
        return transformed_data



