import random
from typing import Union
from typing import List
from typing import Any

from deeplodocus.data.transform.transformer.transformer import Transformer
from deeplodocus.utils.namespace import Namespace


class OneOf(Transformer):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    OneOf class inheriting from Transformer which compute one random transform from the list
    """
    def __init__(self, name: str, mandatory_transforms_start: Union[Namespace, List[dict]], transforms: Union[Namespace, List[dict]], mandatory_transforms_end: Union[Namespace, List[dict]]):
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
        super().__init__(name=name,
                         mandatory_transforms_start=mandatory_transforms_start,
                         transforms=transforms,
                         mandatory_transforms_end=mandatory_transforms_end)

    def transform(self, transformed_data: Any, index: int, augment: bool):
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
        :param augment(bool): Whether to apply non mondatory transforms to the instance


        RETURN:
        -------

        :return transformed_data: The transformed data
        """
        transforms = []
        if self.last_index == index:
            transforms += self.last_transforms

        else: # Get ALL the mandatory transforms + one transform randomly selected
            transforms += self.list_mandatory_transforms_start                                    # Get the mandatory transforms at the start

            if augment is True:
                random_transform_index = random.randint(0, len(self.list_transforms))        # Get a random transform among the ones available in the list
                transforms += self.list_transforms[random_transform_index]                      # Get the one function
            transforms += self.list_mandatory_transforms_end                                    # Get the mandatory transforms at the end


        # Reinitialize the last transforms
        self.last_transforms = []

        # Apply the transforms
        transformed_data = self.apply_transforms(transformed_data, transforms)

        self.last_index = index
        return transformed_data
