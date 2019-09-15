# Python imports
from typing import List
from typing import Any
from typing import Tuple
from typing import Union
import weakref

# Deeplodocus imports
from deeplodocus.utils.notification import Notification
from deeplodocus.data.load.source import Source
from deeplodocus.data.load.source_wrapper import SourceWrapper
from deeplodocus.data.load.source_pointer import SourcePointer
from deeplodocus.utils.generic_utils import get_module
from deeplodocus.data.load.loader import Loader

# Import flags
from deeplodocus.flags import *


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


    """

    def __init__(self,
                 index: int,
                 name: str,
                 dataset: weakref,
                 load_as: str,
                 enable_cache: bool = False,
                 cv_library: Union[str, None, Flag] = DEEP_LIB_OPENCV):

        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize an entry for the Dataset

        PARAMETERS:
        -----------

        :param dataset(weakref): Weak reference to the dataset
        :param

        RETURN:
        -------

        :return: None
        """

        # ID of the entry
        self.index = index

        # Optional name of the Entry
        self.name = name

        # Data type
        self.load_as = load_as

        # Weak reference to the dataset
        self.dataset = dataset

        # Loader
        self.loader = Loader(data_entry=weakref.ref(self),
                             load_as=load_as,
                             cv_library=cv_library)

        # List of sources into the entry
        self.sources = list()

        # Enable cache memory for pointer
        self.enable_cache = enable_cache

        # Cache Memory for pointers
        if self.enable_cache is True:
            self.cache_memory = list()
        else:
            self.cache_memory = None

        self.num_instances = None

    def __getitem__(self, index: int):
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :param index:
        :return:
        """

        # If cache memory enabled, reset the cache
        if self.enable_cache is True:
            self.cache_memory = list()


        # Compute the Source  ID and the Instance ID in the Source
        source_index, instance_index = self.__compute_source_indices(index=index)

        # Get the corresponding source
        s = self.sources[source_index]

        # Get the items from the Source instance
        items, is_loaded, is_transformed = s.__getitem__(instance_index)

        if is_loaded is False:
            items = self.loader.load_from_str(items)

        # If cache memory enabled, store the items in cache
        if self.enable_cache is True:
            self.cache_memory = items

        # The item is either the unique item returned or the desired item of the list
        if isinstance(items, tuple):
            items = items[s.get_instance_id()]

        return items, is_transformed

    def __len__(self) -> Union[int, None]:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the number of raw instances in the Entry

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return self.num_instances (Union[int, None]): The number of raw instances within the Entry (None is unlimited Entry)
        """
        return self.num_instances

    def get_first_item(self) -> Any:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the first item of the Entry instance

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return items (Any): The first item of the entry
        """

        # If cache memory enabled, reset the cache
        if self.enable_cache is True:
            self.cache_memory = list()

        # Get the corresponding source
        s = self.sources[0]

        # Get the items from the Source instance
        items, is_loaded, is_transformed = s.__getitem__(0)

        # If cache memory enabled, store the items in cache
        if self.enable_cache is True:
            self.cache_memory = items

        # The item is either the unique item returned or the desired item of the list
        if isinstance(items, tuple):
            items = items[s.get_instance_id()]

        return items

    def clear_cache_memory(self) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Clear the cache memory

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        self.cache_memory = list()

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

        # Initialize a temporary index and a temporary variable for a previous index
        temp_index = 0
        prev_temp_index = 0

        # For each source
        for i, source in enumerate(self.sources):

            # Add the length of the Source to the temp variable
            temp_index += source.get_num_instances()

            # If the index is smaller than the summed of th Sources, return the index of the Source and the relative index of the instance within this Source
            if index < temp_index:
                return i, index - prev_temp_index

            # Else update the temp value
            prev_temp_index = temp_index

        Notification(DEEP_NOTIF_DEBUG, "Error in computing the source index... Please check the algorithm")

    def get_item_from_cache(self, index: int)-> Any:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get an item stored in the cache of the Entry
        Accessible only by SourcePointer instances

        1) Check the Entry has enabled the cache memory
        2) Check the index is within the range of the cache memory
        3) Get the item from the cache

        PARAMETERS:
        -----------

        :param index(int): Index of the cache we want to access

        RETURN:
        -------

        :return item(Any): The cached item
        """

        # 1) Check if the cache memory is enabled on this Entry
        if self.enable_cache is False:

            # Get info on Entry and Dataset for DEEP_FATAL Notification display
            entry_info = self.get_info()
            dataset_info = self.dataset().get_info()

            # Display Fatal error if cache memory disabled
            Notification(DEEP_NOTIF_FATAL,
                         message="The Entry %s in the Dataset %s does not have cache memory enabled. SourcePointer instances cannot access the data in cache memory." % (entry_info, dataset_info),
                         solutions="Set 'enable_cache' to True for Entry %s in Dataset %s" % (entry_info, dataset_info))

        # 2) Check if the index is within the range of the cache memory
        self.check_cache_size(index)

        # 3) Get the item from the cache memory
        return self.cache_memory[index]

    def generate_sources(self, sources: List[dict]) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Generate the sources
        Does not generate the SourcePointer instances

        PARAMETERS:
        -----------

        :param sources(List[dict]): The configuration of the Source instances to generate

        RETURN:
        -------

        :return: None
        """
        list_sources = []

        # Create sources
        for i, source in enumerate(sources):

            # Get the Source module and add it to the list
            module, origin = get_module(name=source["name"], module=source["module"], browse=DEEP_MODULE_SOURCES)

            # If the source is not a real source
            if issubclass(module, Source) is False:
                # Remove the id from the initial kwargs
                index = source["kwargs"].pop('index', None)

                # Create a source wrapper with the new ID
                s = SourceWrapper(index=index,
                                  name=source["name"],
                                  module=source["module"],
                                  kwargs=source["kwargs"])
            else:
                # If the subclass is a real Source
                s = module(**source["kwargs"])

            # Check the module inherits the generic Source class
            self.check_type_sources(s, i)

            # Add the module to the list of Source instances
            list_sources.append(s)

        # Set the list as the attribute
        self.sources = list_sources

    def calculate_cache_length(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Calculate the length of the cache memory
        Useful to know if the given instance index from a SourcePointer fits within the size of the cache

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return cache_length (int): The length of the cache
        """
        # TODO:
        pass

    def reorder_sources(self) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Reorder the list of Source instances
        SourcePointer instances are generated after the normal Source instances and therefore are appended at the end of the list
        However, a SourcePointer position might be before a normal Source

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        # List of current Source order
        source_order = []

        # Get all the source ID
        for source in self.sources:
            source_order.append(source.get_index())

        # Reorder the sources
        self.sources = [self.sources[i] for i in source_order]

    def is_next_source_pointer(self, index: int) -> bool:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check whether the Source required for a specific index is a SourcePointer instance or not

        PARAMETERS:
        -----------

        :param index (int): Index of a specific item in the Entry

        RETURN:
        -------

        :return is_pointer (bool): Whether the Source instance require for a specific index is a SourcePointer instance
        """

        is_pointer = False

        # Compute the Source ID which will be called
        source_index, _ = self.__compute_source_indices(index=index)

        # Get the specific Source instance
        source = self.sources[source_index]

        # Check if it is a SourcePointer instance
        if isinstance(source, SourcePointer):
            is_pointer = True

        return is_pointer

    def compute_num_raw_instances(self) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Compute the number of raw instances within the Entry

        If any Source instance is unlimited (None):
            => Considered the Entry as unlimited
        Else :
            => Sum the length of all Source instances

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        # Initialize a list which will contain the length of all Source instances
        source_lengths = [None]*len(self.sources)

        # Get the length of each Source
        for i in range(len(self.sources)):
            l = self.sources[i].get_num_instances()

            # If the length is None => The whole Entry is considered as unlimited
            if l is None:
                return None
            # Else we store in memory the length of the Source instance
            else:
                source_lengths[i] = l

        # The length of the Entry is the sum of all the Source instances length
        self.num_instances = sum(source_lengths)



    ############
    # CHECKERS #
    ############

    def check(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check the Entry:
            1) Check the Source instances
            2) Verify the custom Source were correctly checked

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        # Check Source instances
        for i in range(len(self.sources)):
            self.sources[i].check()

        # Very custom Source instances were correctly checked
        for i in range(len(self.sources)):
            self.sources[i].verify_custom_source()



    def check_loader(self):
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
        # Check the Loader
        self.loader.check()

    ############
    # CHECKERS #
    ############

    def check_type_sources(self, s: Any, source_index: int) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check that all the items in self.sources are Source instances

        PARAMETERS:
        -----------

        :param s(Any): A supposed Source instance to check

        RETURN:
        -------

        :return: None
        """
        # Check if the item inherits the generic Source class
        if isinstance(s, Source) is False:

            # Get Dataset and Entry info
            entry_info = self.get_info()
            dataset_info = self.dataset().get_info()

            # Display Error message
            Notification(DEEP_NOTIF_FATAL,
                         "The source item #%i in Entry %s of the Dataset %s does not inherit the generic Source class of Deeplodocus" %(source_index, entry_info, dataset_info),
                         solutions="Make sure the given Source follows the Deeplodocus Source format")

    def check_existing_source(self, index: int):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check if a specific source exists using its index

        PARAMETERS:
        -----------

        :param index(int): The index of the Source instance

        RETURN:
        -------

        :return exists(bool): Whether the Source instance exists or not
        """
        exists = False

        # If the index is smaller than the length of the list of sources then it has to exist
        if index < len(self.sources):
            exists = True

        return exists

    def check_cache_size(self, index: int):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check if the item of a specific index is available in the cache

        PARAMETERS:
        -----------

        :param index(int): Index of the item desired in the cache memory

        RETURN:
        -------

        :return (bool): Whether or not the index is in the cache size range
        """

        # Get the length of the cache memory
        length_cache = len(self.cache_memory)

        # If we have no info on the length of the cache we check the number of output arguments
        if length_cache == 0:
            # Get the real length of the cache memory
            # length_cache = self.calculate_cache_length()
            Notification(DEEP_NOTIF_FATAL, "Cache memory not filled before usage")

        if index < length_cache:
            return True
        else:
            return False

    def check_output_sizes_sources(self, sources_output_sizes: List[int]) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check all the Source instances output the same number of arguments

        PARAMETERS:
        -----------

        :param sources_output_sizes(List[int]): List of output sizes for all Source instances in the Entry

        RETURN:
        -------

        :return: None
        """

        first_output_size = sources_output_sizes[0]

        for outputs_size in sources_output_sizes:
            if first_output_size != outputs_size:

                # Get info on Entry and Dataset for DEEP_FATAL Notification display
                entry_info = self.get_info()
                dataset_info = self.dataset().get_info()

                # Display Fatal Error
                Notification(DEEP_NOTIF_FATAL,
                             "All the Source instance in the Entry %s in Dataset %s do not have the same number of output items" % (entry_info, dataset_info),
                             solutions="Please check each Source instance of Entry %s in Dataset %s and make sure they output the same number of arguments in __getitem__() "% (entry_info, dataset_info))

    ###########
    # GETTERS #
    ###########

    def get_source(self, index: int):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check if a specific source exists
        Get a specific source in the entry

        PARAMETERS:
        -----------

        :param index(int): index of the Source  instance in the Entry instance

        RETURN:
        -------

        :return source(Source): The requested source
        """

        # Check if the source exists
        self.check_existing_source(index)

        # return the requested source
        return self.sources[index]

    def get_sources(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the list of Source instances

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return (List[Source]): The list of Source instances in the Entry
        """
        return self.sources

    def get_ref(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the weak reference to the entry

        PARAMETER:
        ----------

        :param: None

        RETURN:
        -------

        :return self (weakref): The weak reference to the entry
        """
        return weakref.ref(self)

    def get_info(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Format and return info for the entry

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return entry_info(str): Information on the Entry
        """
        # If there is no name for the Entry we use the Entry ID
        if self.name is None:
            entry_info = "#" + str(self.index)
        else:
            entry_info = self.name

        return entry_info

    def get_index(self) -> int:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the index of the Entry in the Dataset

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return self.index(int): The Entry index
        """
        return self.index

    def get_type(self):
        return self.type

    def set_source_pointer_weakref(self, source_id: int, entry_weakref: weakref):
        self.sources[source_id].set_entry_weakref(entry_weakref)