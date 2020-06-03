# Python imports
from typing import Any
from typing import Union
from typing import List

# Deeplodocus imports
from deeplodocus.data.transform.transformer.transformer import Transformer
from deeplodocus.utils.namespace import Namespace


class Sequential(Transformer):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Sequential class inheriting from Transformer which compute the list of transforms sequentially
    """

    def __init__(
            self,
            name: str,
            mandatory_transforms_start: Union[Namespace, List[dict]],
            transforms: Union[Namespace, List[dict]],
            mandatory_transforms_end: Union[Namespace, List[dict]]
    ):
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
        super().__init__(
            name=name,
            mandatory_transforms_start=mandatory_transforms_start,
            transforms=transforms,
            mandatory_transforms_end=mandatory_transforms_end
        )

    def transform(self, transformed_data: Any, index: int, augment: bool) -> Any:
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

        RETURN:
        -------

        :return transformed_data: The transformed data
        """
        transforms = []

        if self.last_index == index:
            transforms += self.last_transforms

        else:
            # Get mandatory transforms + transform
            if augment is True:
                transforms = self.list_mandatory_transforms_start + self.list_transforms + self.list_mandatory_transforms_end
            else:
                transforms = self.list_mandatory_transforms_start + self.list_mandatory_transforms_end

        # Reinitialize the last transforms
        self.last_transforms = []

        # Apply the transforms
        transformed_data = self.apply_transforms(transformed_data, transforms)

        # Update the last index
        self.last_index = index
        return transformed_data
