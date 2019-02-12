# Python imports
import os
from typing import List
from typing import Any
from typing import Union
from typing import Optional
from typing import Tuple
import mimetypes

# Deeplodocus imports
from deeplodocus.utils.notification import Notification
from deeplodocus.utils.generic_utils import sorted_nicely
from deeplodocus.utils.generic_utils import get_int_or_float
from deeplodocus.utils.generic_utils import is_np_array
from deeplodocus.utils.generic_utils import get_corresponding_flag
from deeplodocus.data.source import Source

# Import flags
from deeplodocus.utils.flags.source import *
from deeplodocus.utils.flags.notif import *
from deeplodocus.utils.flags.msg import *
from deeplodocus.utils.flags.load import *
from deeplodocus.utils.flags.entry import *
from deeplodocus.utils.flags.dtype import *
from deeplodocus.utils.flags.flag_lists import DEEP_LIST_DTYPE
from deeplodocus.utils.flags.flag_lists import DEEP_LIST_ENTRY


class Entry(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Entry class
    An Entry instance represents one entry for the model.
    Each entry is composed of one or multiple sources of data combined into one unique list whatever the origin (file, folder, database ...)

    The Entry class manages everything for an entry:
        - Analyses the type of entry
        - Analyses the loading method
        - Join a parent directory if a relative path is given in data
        - Store data in memory if requested by the user

    PUBLIC METHODS:
    ---------------

    :method get_source:
    :method get_source_type:
    :method get_join:
    :method get_data_type:
    :method get_load_method:
    :method get_entry_index:
    method get_entry_type:

    PRIVATE METHODS:
    ----------------

    :method __init__: Initialize an Entry instance
    :method __len__: Get the length of the Entry
    :method __getitem__: Get the selected item
    :method __get_data_from_memory:
    :method __load_online:
    :method __compute_source_indices:
    :method __convert_source_folder_to_file: Convert the folder sources to files for efficiency
    :method __check_sources:
    :method __check_join:
    :method __check_data_type:
    :method __estimate_data_type:
    :method __check_load_method:
    :method __check_entry_type:
    :method __check_folders:
    :method __load_content_in_memory:
    :method __read_folders:
    """

    def __init__(self,
                 sources: Union[str, List[str]],
                 join: Union[str, List[str], None],
                 entry_index: int,
                 entry_type: Union[str, int, Flag],
                 dataset,
                 data_type: Union[str, int, Flag, None] = None,
                 load_method: Union[str, int, Flag, None] = "default"):

        """
        AUTHORS:
        --------

        :author: Alix Leroy


        DESCRIPTION:
        ------------

        Initialize an entry for the dataset

        PARAMETERS:
        -----------

        :param sources:
        :param source_types:
        :param join:
        :param dtype:
        :param load_method:
        :param entry_index (int):
        :param entry_type :

        RETURN:
        -------

        :return: None
        """
        self.dataset = dataset
        self.sources = self.__check_sources(sources=sources, join=join)
        self.data_type = self.__check_data_type(data_type)
        self.load_method = self.__check_load_method(load_method)
        self.entry_index = entry_index
        self.entry_type = self.__check_entry_type(entry_type)

        # Check folders (convert folders to files)
        self.__check_folders()

        # Loading method dependant actions
        # OFFLINE
        if DEEP_LOAD_METHOD_OFFLINE.corresponds(info=load_method):
            self.__load_content_in_memory(self.sources)

        # SEMI ONLINE
        #elif DEEP_LOAD_METHOD_SEMI_ONLINE.corresponds(info=load_method):
        #    self.__load_source_in_memory(self.sources)

        # Evaluate the length of the entry
        self.__len__()

    def __len__(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the length of the entry.
        The length of the entry is computed differently whether the loading method is offline or online.

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return (int): The length of the entry
        """
        self.length = sum([s.__len__(load_method=self.load_method) for s in self.sources])
        return self.length

    def __getitem__(self, index: int) -> Tuple[Any, bool, bool]:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the request item in string format

        PARAMETERS:
        -----------

        :param index (int): The index of the item

        RETURN:
        -------

        :return item (str): The item at the selected index in string format
        """

        item = None
        is_loaded = False
        is_transformed = False

        # Get the corresponding source
        source_index, index_in_source = self.__compute_source_indices(index=index)


        # OFFLINE
        if self.load_method() == DEEP_LOAD_METHOD_OFFLINE():
            # Get the data
            item, _, _ = self.sources[source_index].__getitem__(index_in_source)
            is_loaded = True
            is_transformed = True

        # SEMI-ONLINE
        #elif self.load_method() == DEEP_LOAD_METHOD_SEMI_ONLINE():
        #    item, is_loaded, is_transformed = self.__load_semi_online(index=index_in_source)

        # ONLINE
        elif self.load_method() == DEEP_LOAD_METHOD_ONLINE():
            # Get the data
            item, is_loaded, is_transformed = self.sources[source_index].__getitem__(index_in_source)

        # OTHERS
        else:
            Notification(DEEP_NOTIF_FATAL, "Loading method not implemented : %s" % str(self.load_method.get_description()))

        return item, is_loaded, is_transformed


    def __compute_source_indices(self, index: int) -> Tuple[int, int]:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Compute the source index


        PARAMETERS:
        -----------

        :param index(int): The index of the data to load

        RETURN:
        -------

        :return (Tuple[int, int]): [The index of the source to load from, The index of the instance in the source]
        """

        temp_index = 0
        prev_temp_index = 0

        for i, source in enumerate(self.sources):
            temp_index += source.get_length()

            if index <= temp_index:
                return i, index - prev_temp_index
            prev_temp_index = temp_index

        Notification(DEEP_NOTIF_DEBUG, "Error in computing the source index... Please check the algorithm")



    def __convert_source_folder_to_file(self, source : Source, source_index : int):
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

        item_list = self.__read_folders(source.get_source())

        # generate the absolute path to the file
        filepath = DEEP_ENTRY_BASE_FILE_NAME %(self.dataset.get_name(), self.entry_type.get_name(), self.entry_index, source_index)

        # Create the folders if required
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            for item in item_list:
                f.write("%s\n" % item)

        return filepath

    """
    "
    " CHECKERS
    "
    """

    def __check_sources(self, sources : Union[str, List[str]],  join : Union[str, None, List[Union[str, None]]]) -> List[Source]:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check the source given is a string or a list of strings
        If it is a single source, we transform it to a list of a single element

        PARAMETERS:
        -----------

        :param sources (Union[str, List[str]]) : The source or list of sources
        :param join (Union[str, List[str]) : The list of jointures

        RETURN:
        -------

        :return sources(List[Source]): List of sources
        """
        formatted_sources = []

        # Convert single elements to a list of one element
        if isinstance(sources, list) is False:
            sources = [sources]

        # Check joins
        join = self.__check_join(join=join)

        # Check the number of jointures and sources are equal:
        if len(sources) != len(join):
            Notification(DEEP_NOTIF_FATAL, "The entry %s (index %i) does not have the same amount of sources and jointures" % (self.entry_type.get_description(), self.entry_index))

        # Check all the elements in the list to make sure they are strings
        for i, item in enumerate(sources):
            if isinstance(item, str) is True:
                source = Source(source=item, join =join[i])
                formatted_sources.append(source)
            else:
                Notification(DEEP_NOTIF_FATAL, "The source parameter '%s' in the %i th %s entry is not valid" %(str(item), self.entry_type.get_description(), self.entry_index))

        return formatted_sources

    def __check_join(self, join : Union[str, None, List[Union[str, None]]]):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check the join is a string or a list of strings
        If it is a single source, we transform it to a list of a single element

        PARAMETERS:
        -----------

        :param join (Union[str, List[str]]) : The source or list of sources

        RETURN:
        -------

        :return join(List[str]): List of jointures
        """

        # Convert single elements to a list of one element
        if isinstance(join, list) is False:
            join = [join]

        # Check all the elements in the list to make sure they are strings
        for item in join:
            if isinstance(item, str) is False and item is not None:
                Notification(DEEP_NOTIF_FATAL, "The join parameter '%s' in the %i th %s entry is not valid" %(str(item), self.entry_type.get_description(), self.entry_index))

        return join

    def __check_data_type(self, data_type: Union[str, int, Flag, None]):
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

        :param data_type (Union[str, int, None]: The data type in a raw format given by the user

        RETURN:
        -------

        :return data_type(Flag): The data type of the entry
        """

        if data_type is None:
            instance_example, _, _ = self.__getitem__(index=0)
            # Automatically check the data type
            data_type = self.__estimate_data_type(instance_example)
        else:
            data_type = get_corresponding_flag(flag_list=DEEP_LIST_DTYPE,
                                               info=data_type)
        return data_type

    def __estimate_data_type(self, data: str) -> Flag:
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
            dtypes = []
            # Get all the data type
            for d in data:
                dt = self.__estimate_data_type(d)
                dtypes.append(dt)

            # Check the data types are all the same
            for dt in dtypes:
                if dtypes[0].corresponds(dt) is False:
                    Notification(DEEP_NOTIF_FATAL, "Data type in your sequence of data are not all the same")

            # If all the same then return the data type
            return dtypes[0]

        # If not a list
        else:
            mime = mimetypes.guess_type(data)
            if mime[0] is not None:
                mime = mime[0].split("/")[0]

            # IMAGE
            if mime == "image":
                return DEEP_DTYPE_IMAGE
            # VIDEO
            elif mime == "video":
                return DEEP_DTYPE_VIDEO
            # FLOAT
            elif DEEP_DTYPE_FLOAT.corresponds(get_int_or_float(data)):
                return DEEP_DTYPE_FLOAT
            # INTEGER
            elif DEEP_DTYPE_INTEGER.corresponds(get_int_or_float(data)):
                return DEEP_DTYPE_INTEGER
            # NUMPY ARRAY
            if is_np_array(data) is True:
                return DEEP_DTYPE_NP_ARRAY
            # Type not handled
            else:
                Notification(DEEP_NOTIF_FATAL, DEEP_MSG_DATA_NOT_HANDLED % data)

    @staticmethod
    def __check_load_method(load_method : Union[str, int, Flag, None]) -> Flag:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check if the load method required is valid.
        Check if the load method given is an integer, otherwise convert it to the corresponding flag

        PARAMETERS:
        -----------

        :param load_method (Union[str, int, Flag]): The loading method selected

        RETURN:
        -------

        :return load_method (Flag): The corresponding DEEP_LOAD_METHOD flag
        """

        # TODO : Replace the method by the following line once SEM-ONLINE is correctly implemented
        # return get_corresponding_flag(DEEP_LIST_LOAD_METHOD_HIGH_LEVEL(), info=load_method, fatal=False, default=DEEP_LOAD_METHOD_ONLINE)

        # OFFLINE
        if DEEP_LOAD_METHOD_OFFLINE.corresponds(load_method):
            return DEEP_LOAD_METHOD_OFFLINE

        # SEMI-ONLINE
        elif DEEP_LOAD_METHOD_SEMI_ONLINE.corresponds(load_method):
            Notification(DEEP_NOTIF_FATAL, "Semi online loading not implemented yet")
            return DEEP_LOAD_METHOD_SEMI_ONLINE

        # ONLINE
        elif DEEP_LOAD_METHOD_ONLINE.corresponds(load_method):
            return DEEP_LOAD_METHOD_ONLINE

        # ELSE default (online)
        else:
            Notification(DEEP_NOTIF_WARNING, "The following loading method does not exist : %s, the default online has been selected instead" %str(load_method))
            return DEEP_LOAD_METHOD_ONLINE

    @staticmethod
    def __check_entry_type(entry_type: Union[str, int, Flag]):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check the entry type

        PARAMETERS:
        -----------

        :param entry_type (Union[str, int, Flag]): The raw entry type

        RETURN:
        -------

        :return entry_type(Flag): The entry type
        """
        return get_corresponding_flag(flag_list=DEEP_LIST_ENTRY, info=entry_type)

    def __check_folders(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check if the sources are folders and convert them to files if so

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """

        for i, source in enumerate(self.sources):
            if DEEP_SOURCE_FOLDER.corresponds(source.get_type()):
                folder = source.get_source()
                filepath = self.__convert_source_folder_to_file(source, i)
                source.set_source(filepath)
                source.set_type(DEEP_SOURCE_FILE)

                Notification(DEEP_NOTIF_WARNING, "The folder '%s' has been converted to a file for efficiency at '%s'" %(folder, filepath))


    """
    "
    " MEMORY
    "
    """

    def __load_content_in_memory(self, sources: List[Source]) -> None:
        """
        AUTHORS:
        --------

        author: Alix Leroy

        DESCRIPTION:
        ------------

        List all the data from a file or from a folder

        PARAMETERS:
        -----------

        :param sources (List[Source]): A list of sources

        RETURN:
        -------

        :return None
        """


        # For each source
        for source in sources:

            content = []
            for i in range(source.__len__()):
                data = source.__getitem__(index=i)
                content.append(data)

            source.set_data_in_memory(content=content)

        Notification(DEEP_NOTIF_SUCCESS, "Entry %s %i : Content loaded in memory" % (str(self.entry_type.get_description()), self.entry_index))

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



    """
    "
    " GETTERS
    "
    """

    def get_source(self, index: Optional[int] = None) -> Union[str, List[str]]:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the sources list or a specific source item

        PARAMETERS:
        -----------

        :param index (Optional[int]):  The index of source item selected

        RETURN:
        -------

        :return: the sources list or an item from the source list
        """
        if index is None:
            return [s.get_source() for s in self.sources]
        else:
            return self.sources[index].get_source()

    def get_source_type(self, index: Optional[int] = None) -> Union[Flag, List[Flag]]:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the source types list or a specific source type item

        PARAMETERS:
        -----------

        :param index (Optional[int]):  The index of source type item selected

        RETURN:
        -------

        :return: the source types list or an item from the source type list
        """
        if index is None:
            return [s.get_type() for s in self.sources]
        else:
            return self.sources[index].get_type()

    def get_join(self, index: Optional[int] = None) -> Union[str, List[str]]:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the join list or a specific join item

        PARAMETERS:
        -----------

        :param index (Optional[int]):  The index of join item selected

        RETURN:
        -------

        :return: the join list or an item from the join list
        """
        if index is None:
            return [s.get_join() for s in self.sources]
        else:
            return self.sources[index].get_join()

    """
    "
    " GETTERS
    "
    """


    def get_data_type(self) -> Flag:
        return self.data_type

    def get_load_method(self) -> Flag:
        return self.load_method

    def get_entry_index(self) -> int:
        return self.entry_index

    def get_entry_type(self)-> Flag:
        return self.entry_type

