# Python imports
import os
from typing import List
from typing import Any
from typing import Union
from typing import Optional
import mimetypes

# Deeplodocus imports
from deeplodocus.utils.notification import Notification
from deeplodocus.utils.generic_utils import sorted_nicely
from deeplodocus.utils.file import get_specific_line
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
from deeplodocus.utils.flags.list import DEEP_LIST_DTYPE
from deeplodocus.utils.flags.list import DEEP_LIST_ENTRY


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

    METHODS:
    --------

    public methods :
                    - get_source
                    - get_source_type
                    - get_join
                    - get_entry_index
                    - get_entry_type
                    - get data_type
                    - get load_method
                    - get_data_from_memory

    private methods :
                     -


    """

    def __init__(self, sources: Union[str, List[str]],
                 join: Union[str, List[str], None],
                 entry_index : int,
                 entry_type: Union[str, int, Flag, None],
                 data_type: Union[str, int, Flag, None] = None,
                 load_method : Union[str, int, Flag, None] = "default"):
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

        self.sources = self.__check_sources(sources=sources, join=join)
        self.data_type = self.__check_data_type(data_type)
        self.load_method = self.__check_load_method(load_method)
        self.entry_index = entry_index
        self.entry_type = self.__check_entry_type(entry_type)

        # Loading method dependant actions
        # OFFLINE
        if load_method() == DEEP_LOAD_METHOD_OFFLINE():
            self.data_in_memory = self.__load_content_in_memory(self.sources)

        # SEMI ONLINE
        #elif load_method() == DEEP_LOAD_METHOD_SEMI_ONLINE():
        #    self.data_in_memory = self.__load_source_in_memory(self.sources)

        # ONLINE
        elif load_method() == DEEP_LOAD_METHOD_ONLINE():
            self.__check_folders_for_online_loading()



    def __getitem__(self, index : int) -> str:
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

        item= ""

        # OFFLINE
        if self.load_method() == DEEP_LOAD_METHOD_OFFLINE():
            item = self.__get_data_from_memory(index=index)

        # SEMI-ONLINE
        #elif self.load_method() == DEEP_LOAD_METHOD_SEMI_ONLINE():
        #    item = self.__load_semi_online(index=index)

        # ONLINE
        elif self.load_method() == DEEP_LOAD_METHOD_ONLINE():
            item = self.__load_online(index=index)

        # OTHERS
        else:
            Notification(DEEP_NOTIF_FATAL, "Loading method not implemented : %s" % str(self.load_method.get_description()))

        return item

    def __len__(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the length of the entry.
        The length of the entry is computed differently wether the loading method is offline or online.

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return (int): The length of the entry
        """
        # OFFLINE
        if self.load_method() == DEEP_LOAD_METHOD_OFFLINE():
            return len(self.data_in_memory)

        #elif self.load_method() == DEEP_LOAD_METHOD_SEMI_ONLINE():
        #    # TODO:

        # ONLINE
        elif self.load_method() == DEEP_LOAD_METHOD_ONLINE():
            return sum([l.__len__ for l in self.sources])

    def __convert_source_folder_to_file(self, source : Source):
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
        filepath = DEEP_ENTRY_BASE_FILE_NAME %(self.entry_type, self.entry_index)

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

    def __check_sources(self, sources : Union[str, List[str]], join: Union[str, List[str]]) -> List[Source]:
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

        if isinstance(join, list) is False:
            join = [join]

        # Check the jointures are strings
        for j in join:
            if isinstance(j, str) is False:
                Notification(DEEP_NOTIF_FATAL, "The jointure '%s' is not on a string format" %(str(j)))

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

    def __check_join(self, join : Union[str, List[str]]):
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
            if isinstance(item, str) is False:
                Notification(DEEP_NOTIF_FATAL, "The join parameter '%s' in the %i th %s entry is not valid" %(str(item), self.entry_type.get_description(), self.entry_index))

        return join

    def __check_data_type(self, data_type: Union[str, int, None]):
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
            instance_example = self.__getitem__(index=0)
            # Automatically check the data type
            data_type = self.__estimate_data_type(instance_example)
        else:
            data_type = get_corresponding_flag(flag_list=DEEP_LIST_DTYPE,
                                               info=data_type)
        return data_type

    @staticmethod
    def __estimate_data_type(data) -> int:
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
        mime = mimetypes.guess_type(data)
        if mime[0] is not None:
            mime = mime[0].split("/")[0]

        # IMAGE
        if mime == "image":
            return DEEP_TYPE_IMAGE
        # VIDEO
        elif mime == "video":
            return DEEP_TYPE_VIDEO
        # FLOAT
        elif get_int_or_float(data) == DEEP_TYPE_FLOAT():
            return DEEP_TYPE_FLOAT
        # INTEGER
        elif get_int_or_float(data) == DEEP_TYPE_INTEGER():
            return DEEP_TYPE_INTEGER
        # SEQUENCE
        elif type(data) is list:
            return DEEP_TYPE_SEQUENCE
        # NUMPY ARRAY
        if is_np_array(data) is True:
            return DEEP_TYPE_NP_ARRAY
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
    def __check_entry_type(entry_type : Union[str, int]):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check the entry type

        PARAMETERS:
        -----------

        :param entry_type (Union[str, int]): The raw entry type

        RETURN:
        -------

        :return entry_type(Flag): The entry type
        """

        return get_corresponding_flag(flag_list=DEEP_LIST_ENTRY, info=entry_type)

    def __check_folders_for_online_loading(self):
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
            if source.get_type()() == DEEP_SOURCE_FOLDER():
                folder = source.get_source()
                filepath = self.__convert_source_folder_to_file(source)
                source.set_source(filepath)
                source.set_type(DEEP_SOURCE_FILE)

                Notification(DEEP_NOTIF_WARNING, "The folder '%s' has been converted to a file for efficiency at '%s'" %(folder, filepath))


    """
    "
    " MEMORY
    "
    """

    def __load_content_in_memory(self, sources: List[Source]):
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

        :return content: Content of the entry loaded
        """

        content = []

        # For each source
        for source in sources:

            for i in range(source.__len__()):
                data = source.__getitem__(index=i)
                content.append(data)

        return content


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

    def get_data_type(self) -> Flag:
        return self.data_type

    def get_load_method(self) -> Flag:
        return self.load_method

    def get_entry_index(self) -> int:
        return self.entry_index

    def get_entry_type(self)-> Flag:
        return self.entry_type

    def __get_data_from_memory(self, index: int) -> Any:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get a specific data from the memory

        PARAMETERS:
        -----------

        :param index(int): The index of the data

        RETURN:
        -------

        :return (Any): The requested data
        """
        return self.data_in_memory[index]

    def __load_online(self, index: int):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Load the data directly from the hard drive without accessing the memory
        The method access the source file, folder or database (entry.path) to load the data
        This might be slower because of the multiple reads in files
        This function does not write any content in the file and therefore can be called with a Multiprocessing / Multithreading Dataloader

        PARAMETERS:
        -----------

        :param index (int): The specific index to load

        RETURN:
        -------

        :return: data
        """

        # Get the corresponding source
        source = self.sources[self.__compute_source_index(index=index)]
        stype = source.get_type()

        # FILE
        if stype() == DEEP_SOURCE_FILE():
            data = get_specific_line(filename=source.get_source(),
                                     index=index)

        # BINARY FILE
        # elif source_type == DEEP_SOURCE_BINARY_FILE()
        # TODO: Add binary files

        # FOLDER
        elif stype() == DEEP_SOURCE_FOLDER():
            Notification(DEEP_NOTIF_FATAL, "Load from hard drive with a source folder is supposed to "
                                           "be converted to a source file."
                                           "Please check the documentation to see how to use the Dataset class")

        # DATABASE
        elif stype()() == DEEP_SOURCE_DATABASE():
            Notification(DEEP_NOTIF_FATAL, "Load from hard drive with a source database not implemented yet")

        data = Extractor.extract(data=data, source_type=stype)

    def __compute_source_index(self, index: int):
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

        :return source_index(int): The index of the source to load from
        """

        mod_index = index % self.__len__()

        temp_index = 0

        for i, source in enumerate(self.sources):
            temp_index += source.__len__()

            if mod_index <= temp_index:
                return i

        Notification(DEEP_NOTIF_DEBUG, "Error in computing the source index... Please check the algorithm")
