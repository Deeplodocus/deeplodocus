# Python imports
from typing import Any
from typing import Tuple
from typing import Optional

# Third party imports
import numpy as np

# Deeplodocus imports
from deeplodocus.data.load.source import Source


class LoadableSource(Source):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    LoadableSource class
    A Source class for loading data into memory
    """

    def __init__(self,
                 index: int = -1,
                 is_loaded: bool = False,
                 is_transformed: bool = False,
                 num_instances: Optional[int] = None,
                 load_in_memory: bool = False,
                 instance_id: int = 0):

        super().__init__(index=index,
                         is_loaded=is_loaded,
                         is_transformed=is_transformed,
                         num_instances=num_instances,
                         instance_id=instance_id)

        self.bool_load_in_memory = load_in_memory
        self.memory = list()

        # If we want to load the Source in memory
        if self.bool_load_in_memory is True:
            self.load_offline()

    def __getitem__(self, index: int) -> Tuple[Any, bool, bool]:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get item

        PARAMETERS:
        -----------

        :param index(int):


        :return:
        """
        if self.bool_load_in_memory is True:
            return self.memory[index]

    def load_offline(self) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Load the whole source in memory

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        for i in range(self.num_instances):
            self.add_instance(self.__getitem__(i))

    def add_instance(self, instance: np.array) -> None:
        """
        Authors:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Add an instance to the list stored into memory

        PARAMETERS:
        -----------

        :param instance(np.array): A data instance

        RETURN:
        -------

        :return: None
        """
        self.memory.append(instance)
