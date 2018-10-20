import random

from deeplodocus.data.transformer.transformer import Transformer

class OneOf(Transformer):

    def __init__(self, config):
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

        #transform_name = transform[0]
        transform_method = transform[1]             # Create a generic alias for the transform method
        transform_args = transform[2]                                   # Dictionary of arguments
        transform_data = transform_method(data, **transform_args)

        self.last_transforms[0] = ["", transform_method, transform_args]

        return transform_data