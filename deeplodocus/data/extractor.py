from typing import Any
from typing import Union
from typing import List

# Deeplodocus flags
from deeplodocus.utils.flags.source import *

class Extractor(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Extractor class
    Extract the data to the correct format.
    Currently handles :
                        - images
                        - videos
                        - integers
                        - floats
                        - numpy arrays
                        - sequences (to be tested)
    """

    def __init__(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize the extractor
        """

    def extract(self, data: Any, source_type : Flag) -> Any:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Extract data from a string format to a real format

        PARAMETERS:
        -----------

        :param data:

        RETURN:
        -------

        :return:
        """

        if self.__is_already_extracted(source_type=source_type) is True:
            return data
        else:
            self.__extract(data)


    def __is_already_extracted(self, source_type : Flag):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check if the
        :param source_type:
        :return:
        """

        if source_type() == DEEP_SOURCE_FILE():
            return False
        elif source_type() == DEEP_SOURCE_FOLDER():
            return False
        elif source_type() == DEEP_SOURCE_DATABASE():
            return True
        elif source_type() == DEEP_SOURCE_SERVER():
            return True