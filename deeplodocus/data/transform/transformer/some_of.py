# Python imports
import random
from typing import Any
from typing import Optional
from typing import Union
from typing import List

# Deeplodocus imports
from deeplodocus.data.transform.transformer.transformer import Transformer
from deeplodocus.utils.namespace import Namespace


class SomeOf(Transformer):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    SomeOf class inheriting from Transformer which compute a random number of transforms in the tranforms list.
    The random number is bounded by a min and max
    """

    def __init__(self,
                 name: str,
                 mandatory_transforms_start:  Union[Namespace, List[dict]],
                 transforms:  Union[Namespace, List[dict]],
                 mandatory_transforms_end:  Union[Namespace, List[dict]],
                 num_transformations: Optional[int] = None,
                 num_transformations_min: Optional[int] = None,
                 num_transformations_max: Optional[int] = None) -> None:
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
        super().__init__(name=name,
                         mandatory_transforms_start=mandatory_transforms_start,
                         transforms=transforms,
                         mandatory_transforms_end=mandatory_transforms_end)

        # Compute the number of transformation required
        if num_transformations is None:
            self.num_transformations = None

            if num_transformations_min is None:
                self.num_transformations_min = 1
            else:
                self.num_transformations_min = int(num_transformations_min)

            if num_transformations_max is None:
                self.num_transformations_max = len(self.list_transforms)
            else:
                self.num_transformations_max = int(num_transformations_max)
        else:
            self.num_transformations = num_transformations
            self.num_transformations_min = None
            self.num_transformations_max = None

    def transform(self, transformed_data: Any, index: int, augment: bool) -> Any:
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
        :param index(int): The index of the data
        :param augment(bool): Whether to apply non mandatory transforms to the instance

        RETURN:
        -------

        :return transformed_data: The transformed data
        """
        transforms = []

        if self.last_index == index:
            transforms += self.last_transforms

        else:
            # Add the mandatory transforms at start
            transforms += self.list_mandatory_transforms_start

            # If we want to applied transforms
            if augment is True:
                # If an exact number of transformations is defined
                if self.num_transformations is not None:
                    number_transforms_applied = self.num_transformations

                # Else pick a random number between the boundaries
                else:
                    number_transforms_applied = random.randint(self.num_transformations_min, self.num_transformations_max)

                # Select random transforms from the list
                index_transforms_applied = sorted(random.sample(range(len(self.list_transforms)), number_transforms_applied))      # Sort the list numerically

                # Add the randomly selected transforms to the transform list
                for index in index_transforms_applied:
                    transforms.append(self.list_transforms[index])

            # Apply mandatory transforms at the end
            transforms += self.list_mandatory_transforms_end

        # Reinitialize the last transforms
        self.last_transforms = []

        # Apply the transforms
        transformed_data = self.apply_transforms(transformed_data, transforms)

        # Update the last index
        self.last_index = index
        return transformed_data





