# Python imports
from typing import Optional
from typing import List
from typing import Union
from typing import Any
import numpy as np
import mimetypes
import weakref

# Deeplodocus imports
from  deeplodocus.utils.notification import Notification
from deeplodocus.utils.generic_utils import get_int_or_float
from deeplodocus.utils.generic_utils import is_np_array
from deeplodocus.utils.generic_utils import get_corresponding_flag

# Deeplodocus flags
from deeplodocus.flags import *


class Loader(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Load the unloaded data after being select by the Dataset
    """

    def __init__(self,
                 data_entry: weakref,
                 load_as: Optional[str] = None,
                 cv_library: Union[str, None, Flag] = DEEP_LIB_OPENCV
                 ):

        # Weakref of the Entry instance
        self.data_entry = data_entry

        # Optional type of data to load (Still highly recommended to define it)
        self.load_as = load_as

        # Computer Vision library
        self.warning_video = None
        self.cv_library = None
        self.set_cv_library(cv_library)

        # Checked
        self.checked = False

    def check(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check the Loader

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        # Check the load_as argument
        self.load_as = self.__check_load_as(self.load_as)

        # Set self.checked as True
        self.checked = True

    def __check_load_as(self, load_as: Union[str, int, Flag, None]) -> Flag:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check the data type
        If the data type given is None we try to estimate it (errors can occur with complex types)
        Else we directly get the data type given by the user

        PARAMETERS:
        -----------

        :param load_as (Union[str, int, None]): The data type in a raw format given by the user

        RETURN:
        -------

        :return load_as(Flag): The data type of the entry
        """

        if load_as is None:
            # Get an instance
            instance_example, is_loaded, _ = self.data_entry().__get_first_item()

            if is_loaded is True:
                load_as = None
            else:
                # Automatically check the data type
                load_as = self.__estimate_load_as(instance_example)
        else:
            load_as = get_corresponding_flag(
                flag_list=DEEP_LIST_LOAD_AS,
                info=load_as
            )
        return load_as

    def __estimate_load_as(self, data: str) -> Flag:
        """
        AUTHORS:
        --------

        author: Alix Leroy

        DESCRIPTION:
        ------------

        Find the type of the given data

        PARAMETERS:
        -----------

        :param data: The data to analyze

        RETURN:
        -------

        :return: The integer flag of the corresponding type
        """

        # If we have a list of item, we check that they all contain the same type
        if isinstance(data, list):
            load_as_list = []
            # Get all the data type
            for d in data:
                dt = self.__estimate_load_as(d)
                load_as_list.append(dt)

            # Check the data types are all the same
            for dt in load_as_list:
                if load_as_list[0].corresponds(dt) is False:
                    Notification(DEEP_NOTIF_FATAL, "Data type in your sequence of data are not all the same")

            # If all the same then return the data type
            return load_as_list[0]

        # If not a list
        else:
            mime = mimetypes.guess_type(data)
            if mime[0] is not None:
                mime = mime[0].split("/")[0]

            # IMAGE
            if mime == "image":
                return DEEP_LOAD_AS_IMAGE
            # VIDEO
            elif mime == "video":
                return DEEP_LOAD_AS_VIDEO
            # FLOAT
            elif DEEP_LOAD_AS_FLOAT.corresponds(get_int_or_float(data)):
                return DEEP_LOAD_AS_FLOAT
            # INTEGER
            elif DEEP_LOAD_AS_INTEGER.corresponds(get_int_or_float(data)):
                return DEEP_LOAD_AS_INTEGER
            # NUMPY ARRAY
            if is_np_array(data) is True:
                return DEEP_LOAD_AS_NP_ARRAY
            # Type not handled
            else:
                Notification(DEEP_NOTIF_FATAL, DEEP_MSG_DATA_NOT_HANDLED % data)

    def load_from_str(self, data: Union[str, List[str], Any]) -> Union[Any, List[Any]]:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Load a data from a string format to the actual content
        Loads either one item or a list of items

        PARAMETERS:
        -----------

        :param data(Union[str, List[str]]): The data to transform

        RETURN:
        -------

        :return loaded_data(Union[Any, List[Any]]): The loaded data
        """

        loaded_data = None

        # Make sure the data contains something
        if data is not None:

            # SEQUENCE
            if isinstance(data, list):
                # If data is a sequence we use the function in a recursive fashion
                loaded_data = []
                for d in data:
                    ld = self.load_from_str(data=d)
                    loaded_data.append(ld)

            # IMAGE
            elif DEEP_LOAD_AS_IMAGE.corresponds(self.load_as):
                # Load image
                loaded_data = self.__load_image(data)

            # VIDEO
            elif DEEP_LOAD_AS_VIDEO.corresponds(self.load_as):
                loaded_data = self.__load_video(data)

            # INTEGER
            elif DEEP_LOAD_AS_INTEGER.corresponds(self.load_as):
                loaded_data = int(data)

            # FLOAT NUMBER
            elif DEEP_LOAD_AS_FLOAT.corresponds(self.load_as):
                loaded_data = float(data)

            elif DEEP_LOAD_AS_STRING.corresponds(self.load_as):
                loaded_data = str(data)

            # NUMPY ARRAY
            elif DEEP_LOAD_AS_NP_ARRAY.corresponds(self.load_as):
                loaded_data = np.load(data)

            # LOAD AS GIVEN (unchanged)
            elif DEEP_LOAD_AS_GIVEN.corresponds(self.load_as):
                loaded_data = data

            # Data type not recognized
            else:

                Notification(DEEP_NOTIF_FATAL,
                             "The following data could not be loaded because its type is not recognized : %s.\n"
                             "Please check the documentation online to see the supported types" % data)
        # If the data is None
        else:
            Notification(DEEP_NOTIF_FATAL, DEEP_MSG_DATA_IS_NONE % data)

        return loaded_data


    """
    "
    " DATA LOADERS
    "
    """

    def __load_image(self, image_path: str):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Load the image in the image_path

        PARAMETERS:
        -----------

        :param image_path(str): The path of the image to load

        RETURN:
        -------

        :return: The loaded image
        """
        if DEEP_LIB_OPENCV.corresponds(self.cv_library):
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        elif DEEP_LIB_PIL.corresponds(self.cv_library):
            image = np.array(Image.open(image_path))
        else:
            # Notify the user of invalid cv library
            image = None
            Notification(DEEP_NOTIF_FATAL, DEEP_MSG_CV_LIBRARY_NOT_IMPLEMENTED % self.cv_library.name)

        # Notify the user that the image failed to load
        if image is None:
            Notification(DEEP_NOTIF_FATAL, DEEP_MSG_DATA_CANNOT_LOAD_IMAGE % (self.cv_library.name, image_path))

        # If image is is grayscale add a new dimension
        if image.ndim > 2:
            # If image loaded using openCV, convert to RGB(a)
            if DEEP_LIB_OPENCV.corresponds(self.cv_library):
                image = self.__convert_bgra2rgba(image)
        else:
            image = image[:, :, np.newaxis]

        return image

    @staticmethod
    def __convert_bgra2rgba(image):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Convert BGR(alpha) image to RGB(alpha) image

        PARAMETERS:
        -----------

        :param image: image to convert

        RETURN:
        -------

        :return: a RGB(alpha) image
        """

        # Get the number of channels in the image
        _, _, channels = image.shape

        # Handle BGR and BGR(A) images
        if channels == 3:
            image = image[:, :, (2, 1, 0)]
        elif channels == 4:
            image = image[:, :, (2, 1, 0, 3)]
        return image

    def __load_video(self, video_path: str):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------
        Load a video

        PARAMETERS:
        -----------

        :param video_path->str: absolute path to a video

        RETURN:
        -------

        :return: a list of frame from the video
        """
        self.__throw_warning_video()
        video = []
        # If the computer vision library selected is OpenCV
        if DEEP_LIB_OPENCV.corresponds(self.cv_library):
            # try to load the file
            cap = cv2.VideoCapture(video_path)
            while True:
                _, frame = cap.read()
                if frame is None:
                    break
                video.append(self.__convert_bgra2rgba(frame))
            cap.release()
        else:
            Notification(DEEP_NOTIF_FATAL,
                         "The video could not be loaded because OpenCV is not selected as the Computer Vision library")
        return video

    def __throw_warning_video(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Warn the user of the unstable video mode.

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        if self.warning_video is None:
            Notification(DEEP_NOTIF_WARNING, "The video mode is not fully supported. "
                                             "We deeply suggest you to use sequences of images.")
            self.warning_video = 1

    def set_cv_library(self, cv_library: Flag) -> None:
        """
         AUTHORS:
         --------

         :author: Samuel Westlake
         :author: Alix Leroy

         DESCRIPTION:
         ------------

         Set self.cv_library to the given value and import the corresponding cv library

         PARAMETERS:
         -----------

         :param cv_library: (Flag): The flag of the computer vision library selected

         RETURN:
         -------

         None
         """
        # Set the cv_library argument to the corresponding Flag
        self.cv_library = get_corresponding_flag(flag_list=DEEP_LIST_CV_LIB, info=cv_library)

        # Import globally the required CV library
        self.__import_cv_library(cv_library=cv_library)

    @staticmethod
    def __import_cv_library(cv_library : Flag) -> None:
        """
        AUTHORS:
        --------

        :author: Samuel Westlake
        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Imports either cv2 or PIL.Image dependant on the value of cv_library

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        None
        """
        if DEEP_LIB_OPENCV.corresponds(info=cv_library):
            try:
                global cv2
                import cv2
            except ImportError as e:
                Notification(DEEP_NOTIF_ERROR, str(e))
        elif DEEP_LIB_PIL.corresponds(info=cv_library):
            try:
                global Image
                from PIL import Image
            except ImportError as e:
                Notification(DEEP_NOTIF_ERROR, str(e))
        else:
            Notification(DEEP_NOTIF_ERROR, DEEP_MSG_CV_LIBRARY_NOT_IMPLEMENTED % cv_library)