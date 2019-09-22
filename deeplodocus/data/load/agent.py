# Python imports
from typing import Any
from typing import Tuple
from typing import Optional

# Deeplodocus imports
from deeplodocus.data.load.source import Source


class Agent(Source):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Agent class
    A generic Agent class to get data and send feedback
    """

    def __init__(self, id: int):
        super().__init__(id)

    def __getitem__(self, index: int) -> Tuple[Any, bool, bool]:
        pass

    def __len__(self) -> Optional[int]:
        return None

    def compute_length(self) -> Optional[int]:
        return None