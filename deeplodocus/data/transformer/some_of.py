# Python imports
import random
from typing import Any

# Deeplodocus imports
from deeplodocus.data.transformer.transformer import Transformer


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

    def __init__(self, name, mandatory_transforms, transforms, number_transformations=None, number_transformations_min=None, num_transformations_max=None) -> None:
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
        Transformer.__init__(self, name, mandatory_transforms, transforms)

        # Compute the number of transformation required
        if number_transformations is None :
            self.number_transformation = None

            if number_transformations_min is None:
                self.number_transformations_min = 1
            else:
                self.number_transformations_min = int(number_transformations_min)

            if num_transformations_max is None:
                self.number_transformations_max = len(self.list_transforms)
            else:
                self.number_transformations_max = int(num_transformations_max)
        else:
            self.number_transformation = number_transformations
            self.number_transformation_min = None
            self.number_num_transformations_max = None

    def transform(self, transformed_data: Any, index: int) -> Any:
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
            transforms += self.last_transforms

        else:
            # Add the mandatory transforms
            transforms += self.list_mandatory_transforms

            # If an exact number of transformations is defined
            if self.__number_transformation is not None:
                number_transforms_applied = self.number_transformation

            # Else pick a random number between the boundaries
            else:
                number_transforms_applied = random.randint(self.number_transformations_min, self.number_transformations_max)

            # Select random transforms from the list
            index_transforms_applied = random.sample(range(len(self.list_transforms)), number_transforms_applied ).sort()       # Sort the list numerically

            # Add the randomly selected transforms to the transform list
            for index in index_transforms_applied:
                transforms.append(self.list_transforms[index])

        # Reinitialize the last transforms
        self.last_transforms = []

        # Apply the transforms
        transformed_data = self.apply_transforms(transformed_data, transforms)

        # Update the last index
        self.last_index = index
        return transformed_data





