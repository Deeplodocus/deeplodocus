# Python imports
from typing import Any
from typing import Tuple
from typing import Optional
from typing import List

# Deeplodocus imports
from deeplodocus.data.load.source import Source
from deeplodocus.utils.generic_utils import get_module


class SourceWrapper(Source):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    SourceWrapper class
    """

    def __init__(self,
                 name: str,
                 module: str,
                 kwargs: dict,
                 index: int = -1,
                 is_loaded: bool = True,
                 is_transformed: bool = False,
                 num_instances: Optional[int] = None,
                 instance_id: int = 0,
                 instance_indices: Optional[List[int]] = None):

        super().__init__(index=index,
                         is_loaded=is_loaded,
                         is_transformed=is_transformed,
                         num_instances=num_instances,
                         instance_id=instance_id)

        # Module wrapped and its origin
        module, self.origin = get_module(
            module=module,
            name=name
        )
        # Load module
        self.module = module(**kwargs)

        # Index of the desired source
        self.instance_indices = instance_indices

    def __getitem__(self, index: int) -> Tuple[Any, bool, bool]:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get an item from the cache memory of the selected Entry

        PARAMETERS:
        -----------

        :param index:

        RETURN:
        -------

        :return item (Tuple[Any, bool, bool]):
        """
        # Get the items from the wrapped module
        items = self.module.__getitem__(index)

        # If some specific items need to be loaded
        if self.instance_indices is not None:
            items = self.__select_items(items)

        # Return the items, is_loaded and is_transformed
        return items, self.is_loaded, self.is_transformed

    def __select_items(self, items: List[Any]):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Select a list of items within an existing list of items

        PARAMETERS:
        -----------

        :param items (List[Any]): List of items to pick from

        RETURN:
        -------

        :return selected_items (List[Any]): The list of selected items
        """
        selected_items = []

        for index in self.instance_indices:
            selected_items.append(items[index])

        return selected_items

    def compute_length(self) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Compute the length of the SourcePointer instance by getting the Source instance and getting its length

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return (int): Return the length of the wrapped source
        """
        # Compute the length of the wrapped module
        return self.module.__len__()
