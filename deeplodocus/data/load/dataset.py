# Python import
from typing import List
from typing import Optional
from typing import Any
from typing import Tuple
from typing import Union
import weakref
import numpy as np
import random

# Deeplodocus imports
from deeplodocus.data.load.data_entry import Entry
from deeplodocus.utils.notification import Notification
from deeplodocus.data.load.source_pointer import SourcePointer
from deeplodocus.data.load.pipeline_entry import PipelineEntry
from deeplodocus.data.transform.transform_manager import TransformManager
from deeplodocus.utils.generic_utils import get_corresponding_flag

# Deeplodocus flags
from deeplodocus.flags import *
from deeplodocus.flags.flag_lists import DEEP_LIST_ENTRY


class Dataset(object):
    """
    AUTHORS:
    --------

    :author : Alix Leroy

    DESCRIPTION:
    ------------


    """

    def __init__(self,
                 name: str,
                 entries: List[dict],
                 num_instances: int,
                 transform_manager: Optional[TransformManager],
                 use_raw_data: bool = True,
                 ):

        # Name of the Dataset
        self.name = name

        # List containing the Entry instances
        self.entries = list()
        self.__generate_entries(entries=entries)

        # List containing the PipelineEntry instances
        self.pipeline_entries = list()
        self.__generate_pipeline_entries(entries=entries)

        # Number of raw instances
        self.number_raw_instances = self.__calculate_number_raw_instances()

        # Length of the Dataset
        self.length = self.__compute_length(desired_length=num_instances,
                                            num_raw_instances=self.number_raw_instances)

        # List of items indices
        self.item_order = np.arange(self.length)

        # Whether we want to use raw data or only transformed data
        self.use_raw_data = use_raw_data

        # TransformManager
        self.transform_manager = transform_manager

    def __getitem__(self, index: int):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the item with the corresponding index
        Make sure to load data from Entry instances which do not call SourcePointer instances:
        1) Get a temporary list of Entry instances
            - Reordered list of Entry instances, the Entry instances calling SourcePointer instances are at the end of the list
        2) Get the data from the reordered Entry instances and set them to the right location into the returning list

        PARAMETERS:
        -----------

        :param index (int): Index of the desired instance

        RETURN:
        -------

        :return item(List[Any]): The list of items at the desired index in each Entry instance
        """

        # If the dataset is not unlimited
        if self.length is not None:
            # If the index given is too big => Error
            if index >= self.length:
                Notification(DEEP_NOTIF_FATAL, "The requested instance is too big compared to the size of the Dataset : " + str(index))
            # Else we get the random generated index
            else:
                index = self.item_order[index]

            # If we ask for a not existing index we use the modulo and consider the data to have to be augmented
            if index >= self.number_raw_instances:
                augment = True
            # If we ask for a raw data, augment it only if required by the user
            else:
                augment = not self.use_raw_data
        else:
            index = 0
            augment = True

        # Load items
        items, are_transformed = self.__load_from_entries(index)

        # Transform items
        if self.transform_manager is not None:
            items = self.__transform(index=index,
                                     items=items,
                                     are_transformed=are_transformed,
                                     augment=augment)

        # Format
        items = self.__format(items)

        # Get Inputs, Labels, Additional Data
        inputs, labels, additional_data = self.__split_data_by_entry_type(items)

        return inputs, labels, additional_data

    def __len__(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the number of instances in the Dataset

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return self.num_instances (Union[int, None]): The number of instances within the Dataset (None is unlimited Entry)
        """
        return self.length

    def shuffle(self, method: Flag) -> None:
        """
        AUTHORS:
        --------

        author: Alix Leroy

        DESCRIPTION:
        ------------

        Shuffle the dataframe containing the data

        PARAMETERS:
        -----------

        :param method: (Flag): The shuffling method Flag

        RETURN:
        -------

        :return: None
        """
        # ALL DATASET
        if DEEP_SHUFFLE_ALL.corresponds(info=method):
            self.item_order = np.random.randint(0, high=self.length, size=(self.length,))
            Notification(DEEP_NOTIF_INFO, DEEP_MSG_SHUFFLE_COMPLETE % method.name)

        # NONE
        elif DEEP_SHUFFLE_NONE.corresponds(info=method):
            pass

        # BATCHES
        elif DEEP_SHUFFLE_BATCHES.corresponds(info=method):
            Notification(DEEP_NOTIF_ERROR, "Batch shuffling not implemented yet.")

        # RANDOM PICK
        elif DEEP_SHUFFLE_RANDOM_PICK.corresponds(info=method):
            self.item_order = random.sample(range(0, self.number_raw_instances), self.length)
            Notification(DEEP_NOTIF_INFO, DEEP_MSG_SHUFFLE_COMPLETE % method.name)

        # WRONG FLAG
        else:
            Notification(DEEP_NOTIF_ERROR, DEEP_MSG_SHUFFLE_NOT_FOUND % method.name)

        # Reset the TransformManager
        self.reset()

    def reset(self) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------
        Reset the transform_manager

        PARAMETERS:
        -----------
        None

        RETURN:
        -------
        :return: None
        """
        if self.transform_manager is not None:
            self.transform_manager.reset()

    def __load_from_entries(self, index):

        # Initialize an empty list of N items (N = number of entries) for storing the items
        items = [None for _ in range(len(self.entries))]

        # Initialize an empty list of N items (N = number of entries) for storing if items are transformed
        are_transformed = [False for _ in range(len(self.entries))]

        # Get a temporary order of Entry instances in order to get the item from SourcePointer after normal Source instances
        temp_order = self.__generate_temporary_entries_order(index=index)

        # For each entry, get the data
        for i in range(len(self.entries)):

            # Get the temporary index
            temp_index = temp_order[i]

            # Get entry of the temporary index
            entry = self.entries[temp_index]

            # Get instance
            instance, is_transformed = entry.__getitem__(index)

            # Add instance to the list at the right location
            items[temp_index] = instance
            are_transformed[temp_index] = is_transformed

        return items, are_transformed

    def __transform(self, index: int, items: List[Any], are_transformed: List[bool], augment: bool):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Format the data

        PARAMETERS:
        -----------

        :param index (int): The index of the instance
        :param items (List[Any]): The data to transform
        :param are_transformed (List[bool]): Whether the instances were transformed (1 item per Entry)
        :param augment (bool): Whether we should perform a transformation to the item

        RETURN:
        -------

        :return items (Any): The transformed data
        """
        # For each item check if they have to be transformed
        for i, item in enumerate(items):

            # If not transformed => Call the TransformManager
            if are_transformed[i] is False:
                items[i] = self.transform_manager.transform(data=item,
                                                            entry=self.pipeline_entries[i],
                                                            index=index,
                                                            augment=augment)
        return items

    def __format(self, items: List[Any]) -> List[Any]:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Format the data

        PARAMETERS:
        -----------

        :param items (Any): The data to format

        RETURN:
        -------

        :return items (Any): The formatted data
        """
        for i, pipeline_entry in enumerate(self.pipeline_entries):
            items[i] = pipeline_entry.format(items[i])

        return items

    def __split_data_by_entry_type(self, items: List[Any]) -> Tuple[List[Any], List[Any], List[Any]]:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Format the data

        PARAMETERS:
        -----------

        :param items (Any): The data to format

        RETURN:
        -------

        :return inputs (List[Any]): The list of inputs
        :return labels (List[Any]): The list of labels
        :return additional_data (List[Any]): The list of additional data
        """
        # Initialize lists which will store the input, label and additional_data items
        inputs = list()
        labels = list()
        additional_data = list()

        # We redirect the item to its corresponding PipelineEntry
        for i, pipeline_entry in enumerate(self.pipeline_entries):

            # INPUTS
            if DEEP_ENTRY_INPUT.corresponds(pipeline_entry.get_entry_type()):
                inputs.append(items[i])

            # LABELS
            elif DEEP_ENTRY_LABEL.corresponds(pipeline_entry.get_entry_type()):
                labels.append(items[i])

            # ADDITIONAL DATA
            elif DEEP_ENTRY_ADDITIONAL_DATA.corresponds(pipeline_entry.get_entry_type()):
                additional_data.append(items[i])

        return inputs, labels, additional_data

    def __generate_entries(self, entries: List[dict]) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Generate entries for the dataset

        PARAMETERS:
        -----------

        :param entries(dict): The entries configuration

        RETURN:
        -------

        :return: None
        """
        # Add indices for Entry instances and Source instances
        entries = self.__generate_entry_and_source_indices(entries)

        # Generate the entries
        self.__generate_entries_instances(entries)

        # Does generate the normal Source instances
        # Does NOT generate the SourcePointer instances
        self.__generate_sources(entries)

        # Generate the SourcePointer instances
        self.__generate_source_pointers()

        # Reorder Source instances (because SourcePointer may no be the last Source of the Entry instances)
        # Not needed for now
        # self.reorder_sources()

        # Check entries
        self.__check_entries()

    def __generate_pipeline_entries(self, entries: List[dict]) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Generate the PipelineEntry instances

        PARAMETERS:
        -----------

        :param entries:

        RETURN:
        -------

        :return:
        """
        # Initialize counter to know how many entry of a specific type were created
        inputs = 0
        labels = 0
        additional_data = 0

        # List to store PipelineEntry instances
        generated_pipeline_entries = list()

        # Weak reference to this current Dataset
        weakref_dataset = weakref.ref(self)


        # Generate every single PipelineEntry
        for i, entry in enumerate(entries):

            entry_type = self.__check_entry_type(entries[i]["type"])

            if DEEP_ENTRY_INPUT.corresponds(entry_type):
                entry_type_index = inputs
            elif DEEP_ENTRY_LABEL.corresponds(entry_type):
                entry_type_index = labels
            elif DEEP_ENTRY_ADDITIONAL_DATA.corresponds(entry_type):
                entry_type_index = additional_data
            else:
                Notification(DEEP_NOTIF_FATAL, "The following entry type does not exist")

            # Create new PipelineEntry
            pe = PipelineEntry(index=entries[i]["index"],
                               load_as=entries[i]["load_as"],
                               move_axis=entries[i]["move_axis"],
                               entry_type=entry_type,
                               dataset=weakref_dataset,
                               entry_type_index=entry_type_index)

            if DEEP_ENTRY_INPUT.corresponds(entry_type):
                inputs += 1
            elif DEEP_ENTRY_LABEL.corresponds(entry_type):
                labels += 1
            elif DEEP_ENTRY_ADDITIONAL_DATA.corresponds(entry_type):
                additional_data += 1
            else:
                Notification(DEEP_NOTIF_FATAL, "The following entry type does not exist")

            # Add the entry to the list
            generated_pipeline_entries.append(pe)

        # Change attribute value
        self.pipeline_entries = generated_pipeline_entries

    def __generate_entries_instances(self, entries: List[dict]) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Generate empty Entry instances

        PARAMETERS:
        -----------

        :param entries (List[dict]): The entries configuration in the Dataset

        RETURN:
        -------

        :return: None
        """

        # List to store Entry instances
        generated_entries = list()

        # Weak reference to this current Dataset
        weakref_dataset = weakref.ref(self)

        # Generate every single Entry (empty)
        for i, entry in enumerate(entries):

            # Create new Entry
            e = Entry(index=entries[i]["index"],
                      name=entries[i]["name"],
                      data_type=entries[i]["data_type"],
                      dataset=weakref_dataset,
                      enable_cache=entries[i]["enable_cache"])

            # Add the entry to the list
            generated_entries.append(e)

        # Change attribute value
        self.entries = generated_entries

    def __generate_sources(self, entries: List[dict]) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Generate the Source instances for each Entry of the Dataset

        PARAMETERS:
        -----------

        :param entries (List[dict]): Configuration of the Entry instances in the Dataset

        RETURN:
        -------

        :return: None
        """
        # For each Entry, generate all the Source instances
        # SourcePointer instance are generated but not filled.
        # Weakref for SourcePointer instances are captured later in the generation process
        for i, _ in enumerate(self.entries):
            self.entries[i].generate_sources(entries[i]["sources"])

    def __generate_source_pointers(self) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Generate the SourcePointer instances in the current DataSet
        This has to be done after generating all the other Source instances

        PARAMETERS:
        -----------

        :param entries (List[Entry]):

        RETURN:
        -------
        
        :return: None
        """
        # For each Entry
        for i, entry in enumerate(self.entries):

            # Check each source in the entry
            for j, source in enumerate(entry.get_sources()):

                # If the source is a SourcePointer instance
                if isinstance(source, SourcePointer):

                    # Get the entry index it points to and check the Entry exists
                    entry_index = self.get_entry_index_from_source_pointer(source, entry)

                    # Check the instance ID desired is outputted
                    # TODO: Find a way to extract the number of output arguments from a module
                    # TODO: Maybe we could just check the first instance ?

                    # Get the weakref of the entry it points to
                    entry_weakref = self.entries[entry_index].get_ref()

                    # Send the Entry weakref to the SourcePointer
                    self.entries[i].set_source_pointer_weakref(source_id=j, entry_weakref=entry_weakref)
                    #source.set_entry_weakref(entry_weakref)

    @staticmethod
    def __generate_entry_and_source_indices(entries: List[dict]) -> List[dict]:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Generate an index for every single Entry and Source of the Dataset

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return entries (dict): The updated dataset config with new indices
        """
        # Initialize entry indices
        entry_index = 0

        # Initialize the source indices
        source_index = 0

        # For each Entry we create a new index
        for i, entry in enumerate(entries):
            entries[i]["index"] = entry_index
            entry_index += 1

            # For each Source of each Entry we create a new index
            for j, source in enumerate(entry["sources"]):
                entries[i]["sources"][j]["kwargs"]["index"] = source_index
                source_index += 1

        return entries

    def __generate_temporary_entries_order(self, index: int) -> List[int]:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Generate a list of Entry indices where all the Entry requiring to call a SourcePointer are at the end

        PARAMETERS:
        -----------

        :param index (int): Index of the instances we want to load

        RETURN:
        -------

        :return temp_order (List[int]): The temporary order of Entry indices
        """

        # Initialize list for normal and pointer Source instances
        is_normal = []
        is_pointer = []

        # For each Entry instance
        for entry in self.entries:
            if entry.is_next_source_pointer(index=index) is True:
                is_pointer.append(entry.get_index())
            else:
                is_normal.append(entry.get_index())

        # Concatenate the two lists
        # Normal Source instances first
        # Then all the SourcePointer instances
        temp_order = is_normal + is_pointer

        return temp_order

    def __reorder_sources(self) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Reorder the list of Source instances inside every single Entry
        SourcePointer instances are generated after the normal Source instances and therefore are appended at the end of the list
        However, a SourcePointer position might be before a normal Source

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        # For each Entry instance we reorder the sources
        for entry in self.entries:
            entry.reorder_sources()

    def __calculate_number_raw_instances(self) -> int:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Calculate the theoretical number of instances in each epoch
        The first given file/folder stands as the frame to count

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return num_raw_instances (int): theoretical number of instances in each epoch
        """
        # Calculate for the first entry
        try:
            num_raw_instances = self.entries[0].__len__()
        except IndexError as e:
            Notification(
                DEEP_NOTIF_FATAL,
                "IndexError : %s : %s" % (str(e), DEEP_MSG_DATA_INDEX_ERROR % self.name),
                solutions=[
                    DEEP_MSG_DATA_INDEX_ERROR_SOLUTION_1 % self.name,
                    DEEP_MSG_DATA_INDEX_ERROR_SOLUTION_2 % self.name
                ]
            )

        # For each entry check if the number of raw instances is the same as the first input
        for index, entry in enumerate(self.entries):
            n = entry.__len__()

            if n != num_raw_instances:
                Notification(DEEP_NOTIF_FATAL, "Number of instances in " + str(self.pipeline_entries[0].get_entry_type()) +
                             "-" + str(self.pipeline_entries[0].get_entry_index()) + " and " + str(self.pipeline_entries[index].get_entry_type()) +
                             "-" + str(self.pipeline_entries[index].get_entry_index()) + " do not match.")
        return num_raw_instances

    @staticmethod
    def __compute_length(desired_length: int, num_raw_instances: int) -> int:
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Calculate the length of the dataset

        PARAMETERS:
        -----------

        :param desired_length(int): The desired number of instances
        :param num_raw_instances(int): The actual number of instance in the sources

        RETURN:
        -------

        :return (int): The length of the dataset
        """

        if desired_length is None:
            Notification(DEEP_NOTIF_INFO, DEEP_MSG_DATA_NO_LENGTH % num_raw_instances)
            return num_raw_instances
        else:
            if desired_length > num_raw_instances:
                Notification(DEEP_NOTIF_INFO, DEEP_MSG_DATA_GREATER % (desired_length, num_raw_instances))
                return desired_length
            elif desired_length < num_raw_instances:
                Notification(DEEP_NOTIF_WARNING, DEEP_MSG_DATA_SHORTER % (num_raw_instances, desired_length))
                return desired_length
            else:
                Notification(DEEP_NOTIF_INFO, DEEP_MSG_DATA_LENGTH % num_raw_instances)
                return desired_length

    ############
    # CHECKERS #
    ############

    @staticmethod
    def __check_entry_type(entry_type: Union[str, int, Flag]) -> Flag:
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

    def __check_entries(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        1) Check Entry instances.
            1.1) Check Source instance
            1.2) Very custom source instances
            1.3) Check the Loader.

        2) Clear the cache of all Entry instances (required after checking the Loader)
        3) Compute number of raw instances in each Entry

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        # Check each Entry
        for i in range(len(self.entries)):
            # Check the loader
            self.entries[i].check()
        """
        Check  every Entry instance
        Every entry will then check every instance and the Loader instance
        Require to get a temporary order with Entry calling SourcePointer as last
        The Loader might need to check the data type and loads the first item to check it if required
        NOTE: Entry instance have to be check before checking the Loader
        """

        # Get a temporary order in order to check loaders on the first item
        temp_order = self.__generate_temporary_entries_order(0)

        # Check each Loader in Entry instance
        for i in range(len(self.entries)):
            # Get the temporary index
            temp_index = temp_order[i]
            # Check the loader
            self.entries[temp_index].check_loader()

        for i in range(len(self.entries)):
            # Clear Cache memory in each Entry
            self.entries[i].clear_cache_memory()

            # Compute number of raw instances
            self.entries[i].compute_num_raw_instances()





    ###########
    # GETTERS #
    ###########

    def get_ref(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the weak reference to the dataset

        PARAMETER:
        ----------

        :param: None

        RETURN:
        -------

        :return self (weakref): The weak reference to the dataset
        """
        return weakref.ref(self)

    def get_info(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get information on the Dataset

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return info(str): Information on the Dataset
        """

        return self.name

    def get_entry_index_from_source_pointer(self, source_pointer: SourcePointer, entry: Entry) -> int:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check the entry a source points to does exist

        PARAMETERS:
        -----------

        :param source_pointer(SourcePointer): The SourcePointer to check and extract information from
        :param entry(Entry): Entry instance the SourcePointer belongs to

        RETURN:
        -------

        :return entry_index(int): The index of the Entry instance the SourcePointer points to
        """

        # Get the Entry index
        entry_index = source_pointer.get_entry_index()

        # Check the Entry index
        if entry_index > len(self.entries):
            Notification(DEEP_NOTIF_FATAL,
                         "The SourcePointer %s in the Entry %s of the Dataset %s points to a non existing Entry ID %i" % (source_pointer.get_index(), entry.get_info(), self.get_info(), entry_index),
                         solutions="Make sure the SourcePointer points to an existing Entry.")

        return entry_index

