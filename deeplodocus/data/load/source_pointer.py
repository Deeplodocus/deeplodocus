# Python imports
from typing import Any
from typing import Tuple
from typing import Optional
import weakref

# Deeplodocus imports
from deeplodocus.data.load.source import Source


class SourcePointer(Source):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    SourcePointer class
    A Source class which points to another Entry to get already loaded data.

    Pointing directly to an Entry over a Source offers several advantages:
        1) The Entry offers a higher level of abstraction
        2)

    However it has some drawbacks:
        1) Cannot compute the length of the data we want to point to
        2)
    """

    def __init__(self,
                 index: int = -1,
                 is_loaded: bool = False,
                 is_transformed: bool = False,
                 num_instances: Optional[int] = None,
                 entry_id: int = 0,
                 source_id: int = 0,
                 instance_id: int = 0):

        super().__init__(index=index,
                         is_loaded=is_loaded,
                         is_transformed=is_transformed,
                         num_instances=num_instances,
                         instance_id=instance_id)

        # Entry ID to point to
        self.entry_id = entry_id

        # Source ID to point to
        self.source_id = source_id

        # Weakref of the Entry instance (set later)
        self.weakref_entry = None

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

        :param index (int): Index of the instance requested

        RETURN:
        -------

        :return item (Tuple[Any, bool, bool]):
        """
        return self.weakref_entry().get_item_from_cache(self.instance_id), self.is_loaded, self.is_transformed

    def compute_length(self) -> int:
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

        :return (int): the length of the source
        """
        # Get the desired source
        source = self.weakref_entry().get_source(index=self.source_id)

        # Calculate the length of the pointed source
        return source.__len__()

    def set_entry_weakref(self, weakref_entry: weakref):
        self.weakref_entry = weakref_entry

    def get_entry_index(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the index of the Entry instance the Source is suppose to point to

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return entry_index(int): The index of the Entry instance within the Dataset
        """
        return self.entry_id
