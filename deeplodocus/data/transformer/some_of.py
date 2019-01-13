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
            self.__number_transformation = None

            if number_transformations_min is None:
                self.__number_transformations_min = 1
            else:
                self.__number_transformations_min = int(number_transformations_min)

            if num_transformations_max is None:
                self.__number_transformations_max = len(self.__list_transforms)
            else:
                self.__number_transformations_max = int(num_transformations_max)
        else:
            self.__number_transformation = number_transformations
            self.__number_transformation_min = None
            self.__number_num_transformations_max = None

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

        if self.__last_index == index:
            transforms += self.__last_transforms

        else:
            # Add the mandatory transforms
            transforms += self.__list_mandatory_transforms

            # If an exact number of transformations is defined
            if self.__number_transformation is not None:
                number_transforms_applied = self.__number_transformation

            # Else pick a random number between the boundaries
            else:
                number_transforms_applied = random.randint(self.__number_transformations_min, self.__number_transformations_max)

            # Select random transforms from the list
            index_transforms_applied = random.sample(range(len(self.__list_transforms)), number_transforms_applied ).sort()       # Sort the list numerically

            # Add the randomly selected transforms to the transform list
            for index in index_transforms_applied:
                transforms.append(self.__list_transforms[index])

        # Reinitialize the last transforms
        self.__last_transforms = []

        # Apply the transforms
        transformed_data = self.__apply_transforms(transformed_data, transforms)

        # Update the last index
        self.__last_index = index
        return transformed_data





