# Python imports
import os
from typing import Any
from typing import Tuple
from typing import Union
from typing import List

# Deeplodocus imports
from deeplodocus.utils.notification import Notification
from deeplodocus.utils.file import get_specific_line
from deeplodocus.utils.generic_utils import sorted_nicely
from deeplodocus.utils.file import compute_num_lines

# Deeplodocus flags
from deeplodocus.flags import *
from deeplodocus.flags.notif import *
from deeplodocus.data.load.source import Source


class Folder(Source):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Load a source folder.

    """

    def __init__(self,
                 index=-1,
                 is_loaded=False,
                 is_transformed=False,
                 path="",
                 num_instances=None,
                 instance_id: int = 0
                 ):

        super().__init__(index=index,
                         is_loaded=is_loaded,
                         is_transformed=is_transformed,
                         num_instances=num_instances,
                         instance_id=instance_id,
                         )

        # Save the directory
        self.path = path

        # Check the given path
        self.__check_directory()

        # Save filepath
        self.filepath = self.__convert_source_folder_to_file()

    """
    "
    " LOAD ITEM
    "
    """

    def __getitem__(self, index: int) -> Tuple[Any, bool, bool]:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the item at the selected index

        PARAMETERS:
        -----------

        :param index(int): The index of the selected item

        RETURN:
        -------

        :return data:
        """

        # Check wether the data is loaded and transformed
        is_loaded = self.is_loaded
        is_transformed = self.is_transformed

        # Get the data
        data = get_specific_line(filename=self.filepath,
                                 index=index)

        return data, is_loaded, is_transformed

    def compute_length(self) -> int:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Compute the length using the method corresponding to the source type

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return length(int): The length of the source
        """
        return compute_num_lines(self.filepath)

    def __check_directory(self) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check the given directory exists

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        if not os.path.isdir(self.path):
            Notification(DEEP_NOTIF_FATAL, "The following path is not a Source folder : %s " % self.path)
        else:
            Notification(DEEP_NOTIF_SUCCESS, "Source folder \"%s\" successfully found" % self.path)

    def __convert_source_folder_to_file(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        List the content of a folder into a file

        PARAMETERS:
        -----------

        :param source (Source): A Source instance

        RETURN:
        -------

        :return : None
        """

        item_list = self.__read_folders(self.path)

        # generate the absolute path to the file
        filepath = DEEP_ENTRY_BASE_FILE_NAME % self.index

        # Create the folders if required
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            for item in item_list:
                f.write("%s\n" % item)

        return filepath

    def __read_folders(self, directory: Union[str, List[str]]) -> List[str]:
        """
        AUTHORS:
        --------

        author: Samuel Westlake
        author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the list of paths to every file within the given directories

        PARAMETERS:
        -----------

        :param directory (Union[str, List[str]): path to directories to get paths from

        RETURN:
        -------

        :return (List[str]): list of paths to every file within the given directories

        """
        paths = []
        # For each item in the directory
        for item in os.listdir(directory):
            sub_path = "%s/%s" % (directory, item)
            # If the subpath of the item is a directory we apply the self function recursively
            if os.path.isdir(sub_path):
                paths.extend(self.__read_folders(sub_path))
            # Else we add the path of the file to the list of files
            else:
                paths.extend([sub_path])
        return sorted_nicely(paths)
