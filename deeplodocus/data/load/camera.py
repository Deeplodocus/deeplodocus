# Python imports
from typing import Any
from typing import Tuple
from typing import Optional

# third party libraries
import cv2

# Deeplodocus imports

class Camera(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Camera class
    A Camera source class for loading data from the computer's camera
    """

    def __init__(self,
                 id: int = -1,
                 is_loaded: bool = False,
                 is_transformed: bool = False,
                 camera_id: int = 0):

        self.id = id
        self.is_loaded = is_loaded
        self.is_transformed = is_transformed
        self.cap = cv2.VideoCapture(camera_id)

    def __getitem__(self, index: int) -> Tuple[Any, bool, bool]:
        ret, frame = self.cap.read()
        return frame

    def __len__(self) -> Optional[int]:
        pass

    def compute_length(self) -> Optional[int]:
        return None
