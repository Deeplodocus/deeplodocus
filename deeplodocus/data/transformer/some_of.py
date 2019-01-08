from .transformer import Transformer
import random

class SomeOf(Transformer):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Sequential class inheriting from Transformer which compute a random number of transforms in the tranforms list.
    The random number is bounded by a min and max
    """

    def __init__(self, config):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize a SomeOf transformer inheriting a Transformer

        PARAMETERS:
        -----------

        :param config->Namespace: The config

        RETURN:
        -------

        :return: None
        """
        Transformer.__init__(self, config)

        if hasattr(config, "number_transformations_min"):
            self.number_transformation_min = config.number_transformations_min
        else:
            self.number_transformation_min = 1

        if hasattr(config, "number_transformations_max"):
            self.number_transformation_max = config.number_transformations_max
        else:
            self.number_transformation_max = len(self.list_transforms)

        if hasattr(config, "number_transformations"):
            self.number_transformation = config.number_transformations
        else:
            self.number_transformation = None

    def transform(self, transformed_data, index):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Transform the data using the Some Of transformer

        PARAMETERS:
        -----------

        :param transformed_data: The data to transform
        :param index: The index of the data

        RETURN:
        -------

        :return transformed_data: The transformed data
        """
        transforms = []

        if self.last_index == index:
            transforms = self.last_transforms

        else:
            # Add the mandatory transforms
            transforms += self.list_mandatory_transforms

            if self.number_transformation is not None:
                number_transforms_applied = self.number_transformation
            else:
                number_transforms_applied = random.randint(self.number_transformation_min, self.number_transformation_max)

            index_transforms_applied = random.sample(range(len(self.list_transforms)), number_transforms_applied ).sort()       # Sort the list numerically

            for index in index_transforms_applied:
                transforms.append(self.list_transforms[index])

        # Reinitialize the last transforms
        self.last_transforms = []

        # Apply the transforms
        transformed_data = self.apply_transforms(transformed_data, transforms)

        # Update the last index
        self.last_index = index
        return transformed_data





