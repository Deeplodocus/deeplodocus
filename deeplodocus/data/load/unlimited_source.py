# Python imports
from typing import Any
from typing import Tuple
from typing import Optional

# Third party imports

# Deeplodocus imports
from deeplodocus.data.load.source import Source


class UnlimitedSource(Source):
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
                 instance_id: int = 0):

        super().__init__(index=index,
                         is_loaded=is_loaded,
                         is_transformed=is_transformed,
                         num_instances=num_instances,
                         instance_id=instance_id)


    def __getitem__(self, index: int) -> Tuple[Any, bool, bool]:
        pass

    def compute_length(self) -> Optional[int]:
        return None
