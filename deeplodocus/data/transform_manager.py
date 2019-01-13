from typing import Any
from typing import List
from typing import Optional

# Import transformers
from deeplodocus.data.transformer.one_of import OneOf
from deeplodocus.data.transformer.sequential import Sequential
from deeplodocus.data.transformer.some_of import SomeOf
from deeplodocus.data.transformer.pointer import Pointer
from deeplodocus.data.transformer.no_transformer import NoTransformer

# Deeplodocus imports
from deeplodocus.utils.notification import Notification
from deeplodocus.utils.namespace import Namespace
from deeplodocus.utils.flags.notif import DEEP_NOTIF_FATAL
from deeplodocus.utils.flags.entry import *
from deeplodocus.data.entry import Entry
from deeplodocus.utils.generic_utils import get_corresponding_flag

#Deeplodocus flags
from deeplodocus.utils.flags.flag_lists import DEEP_LIST_TRANSFORMERS
from deeplodocus.utils.flags.transformer import *


class TransformManager(object):
    """
    AUTHORS:
    --------

    author: Alix Leroy

    DESCRIPTION:
    ------------

    A TransformManager class which manages how to transform data
    A TransformManager instance corresponds to one Dataset instance
    The TransformManager class handles three types of transformers:
        - Sequential : applies transformation sequentially as given in the config file
        - OneOf : applies one of the transformation given in the config file
        - SomeOf : applies some of the transformations given in the config file

    The three transformers are gathered under a generic parent Transformer class.

    It is possible to point to another transformer using a pointer.
    This method is very efficient and allows to have exactly the same output on mulitple inputs (e.g. left and right image of stereo vision)
    """

    def __init__(self, name: str, inputs: List[Optional[str]], labels: List[Optional[str]], additional_data: List[Optional[str]]) -> None:
        """
        AUTHORS:
        --------

        author: Alix Leroy and Samuel Westlake

        DESCRIPTION:
        ------------

        Initialize the TransformManager object

        PARAMETERS:
        -----------

        :param name: str: the name of the transformer
        :param inputs: list

        RETURN:
        -------

        None
        """
        self.name = name
        self.list_input_transformers = self.__load_transformers(inputs)
        self.list_label_transformers = self.__load_transformers(labels)
        self.list_additional_data_transformers = self.__load_transformers(additional_data)
        self.summary()

    def transform(self, data : Any, index : int, entry: Entry) -> Any:
        """
        AUTHORS:
        --------

        author: Alix Leroy

        DESCRIPTION:
        ------------

        Transform a data
        Check if the transformer selected is a pointer
        If not a pointer : directly transforms the data
        If a pointer, loads the pointed transformer and then transforms the data

        PARAMETERS:
        -----------

        :param data (Any): The data to transform
        :param index (int): The index of the data to transform
        :param entry (Entry): The entry of the data


        RETURN:
        -------

        :return transformed_data: The transformed data
        """

        list_transformers = None

        #
        # Get the list of transformers corresponding to the type of entry
        #

        # INPUT
        if DEEP_ENTRY_INPUT.corresponds(info=entry.get_entry_type()):
            list_transformers = self.list_input_transformers

        # LABEL
        elif DEEP_ENTRY_LABEL.corresponds(info=entry.get_entry_type()):
            list_transformers = self.list_label_transformers

        # ADDITIONAL DATA
        elif DEEP_ENTRY_ADDITIONAL_DATA.corresponds(info=entry.get_entry_type()):
            list_transformers = self.list_additional_data_transformers

        # WRONG FLAG
        else:
            Notification(DEEP_NOTIF_FATAL, "The following type of entry does not exist : " + str(entry.get_entry_type().get_description()))

        # If it is a NoTransformer instance
        if list_transformers[entry.get_entry_index()].has_transforms() is False:
            return data

        # Check if the transformer points to another transformer
        pointer, pointer_entry_index = list_transformers[entry.get_entry_index()].get_pointer()

        # If we do not point to another transformer, transform directly the data
        if pointer is None:
            transformed_data = list_transformers[entry.get_entry_index()].transform(data, index)

        #
        # If we point to another transformer, load the transformer then transform the data
        #
        else:

            # INPUT
            if DEEP_ENTRY_INPUT.corresponds(pointer):
                list_transformers = self.list_input_transformers

            # LABEL
            elif DEEP_ENTRY_LABEL.corresponds(pointer):
                list_transformers = self.list_label_transformers

            # ADDITIONAL DATA
            elif DEEP_ENTRY_ADDITIONAL_DATA.corresponds(pointer):
                list_transformers = self.list_additional_data_transformers

            # WRONG FLAG
            else:
                Notification(DEEP_NOTIF_FATAL, "The following type of pointer does not exist : " + str(pointer))

            # Transform the data with the freshly loaded transformer
            transformed_data = list_transformers[pointer_entry_index].transform(data, index)

        return transformed_data

    def reset(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Reset all the transformers (avoids pointers and None)

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """

        # Inputs
        for transformer in self.list_input_transformers:
            if transformer is not None and isinstance(transformer, Pointer) is False:
                transformer.reset()

        # Labels
        for transformer in self.list_label_transformers:
            if transformer is not None and isinstance(transformer, Pointer) is False:
                transformer.reset()

        # Additional data
        for transformer in self.list_additional_data_transformers:
            if transformer is not None and isinstance(transformer, Pointer) is False:
                transformer.reset()

    def summary(self):
        """
        AUTHORS:
        --------

        author: Alix Leroy

        DESCRIPTION:
        ------------

        Print the summary of the TransformManager

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        None
        """

        # Inputs
        for transformer in self.list_input_transformers:
            if transformer is not None:
                transformer.summary()

        # Labels
        for transformer in self.list_label_transformers:
            if transformer is not None:
                transformer.summary()

        # Additional data
        for transformer in self.list_additional_data_transformers:
            if transformer is not None:
                transformer.summary()

    def __load_transformers(self, entries):
        """
        CONTRIBUTORS:
        --------

        author: Alix Leroy

        DESCRIPTION:
        ------------

        Load the transformers

        PARAMETERS:
        -----------

        :param entries: List of transformers configs

        RETURN:
        -------

        :return transformers_list: The list of the transformers corresponding to the entries
        """
        entries = entries if isinstance(entries, list) else [entries]
        return [self.__create_transformer(config_entry=entry) for entry in entries]

    def __create_transformer(self, config_entry):
        """
        CONTRIBUTORS:
        -------------

        Creator : Alix Leroy

        DESCRIPTION:
        ------------

        Create the adequate transformer

        PARAMETERS:
        -----------

        :param config: The transformer config
        :param pointer-> bool : Whether or not the transformer points to another transformer

        RETURN:
        -------

        :return transformer: The created transformer
        """
        transformer = None

        # NONE
        if config_entry is None:
            transformer = NoTransformer()

        # POINTER
        elif self.__is_pointer(config_entry) is True:
            transformer = Pointer(config_entry)     # Generic Transformer as a pointer

        # TRANSFORMER
        else:
            config = Namespace(config_entry)

            # Check if a method is given by the user
            if config.check("method", None) is False:
                Notification(DEEP_NOTIF_FATAL, "The following transformer does not have any method specified : " + str(config_entry))

            # Get the corresponding flag
            flag = get_corresponding_flag(flag_list=DEEP_LIST_TRANSFORMERS, info=config.method, fatal = False)

            # Remove the method from the config
            delattr(config, 'method')

            #
            # Create the corresponding Transformer
            #

            # SEQUENTIAL
            if DEEP_TRANSFORMER_SEQUENTIAL.corresponds(flag):
                transformer = Sequential(**config.get())
            # ONE OF
            elif DEEP_TRANSFORMER_ONE_OF.corresponds(flag):
                transformer = OneOf(**config.get())
            # SOME OF
            elif DEEP_TRANSFORMER_SOME_OF.corresponds(flag):
                SomeOf(**config.get())
            # If the method does not exist
            else:
                Notification(DEEP_NOTIF_FATAL, "The following transformation method does not exist : " + str(config.method))

        return transformer

    @staticmethod
    def __is_pointer(source_path: str) -> bool:
        """
        AUTHORS:
        --------

        author: Alix Leroy

        DESCRIPTION:
        ------------

        Check whether or not the source path is a pointer to another transformer (Additional checks are made when creating the pointer itself)

        PARAMETERS:
        -----------

        :param source_path: Source path to the transformer

        RETURN:
        -------

        :return->bool : Whether the source path is a pointer to another transformer
        """

        if str(source_path)[0] == "*":
            return True
        else:
            return False


