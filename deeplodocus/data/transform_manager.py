import time

# Import transformers
from deeplodocus.data.transformer.one_of import OneOf
from deeplodocus.data.transformer.sequential import Sequential
from deeplodocus.data.transformer.some_of import SomeOf
from deeplodocus.data.transformer.pointer import Pointer

from deeplodocus.utils.notification import Notification
from deeplodocus.utils.namespace import Namespace
from deeplodocus.utils.flags.notif import DEEP_NOTIF_FATAL
from deeplodocus.utils.flags.entry import *


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

    def __init__(self, name, inputs, labels, additional_data=None) -> None:
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

    def transform(self, data, index, type_data, entry_type, entry_num):
        """
        AUTHORS:
        --------

        author: Alix Leroy

        DESCRIPTION:
        ------------

        Transform a data

        PARAMETERS:
        -----------

        :param data: The data to transform
        :param index: The index of the data to transform
        :param type_data: Type of data to transform (image, video, sound, ...)
        :param entry_type: Type of entry (input, label, additional_data)
        :param entry_num: Number of the entry (input1, input2, ...) (useful for sequences)

        RETURN:
        -------

        :return transformed_data: The transformed data
        """

        list_transformers = None

        # Get the list of transformers corresponding to the type of entry
        if entry_type == DEEP_ENTRY_INPUT:
            list_transformers = self.list_input_transformers
        elif entry_type == DEEP_ENTRY_LABEL:
            list_transformers = self.list_label_transformers
        elif entry_type == DEEP_ENTRY_ADDITIONAL_DATA:
            list_transformers = self.list_additional_data_transformers
        else:
            Notification(DEEP_NOTIF_FATAL, "The following type of transformer does not exist : " + str (entry_type))


        # Check if the transformer points to another transformer
        pointer = list_transformers[entry_num].get_pointer()

        # If we do not point to another transformer, transform directly the data
        if pointer is None:
        # Transform
            transformed_data = list_transformers[entry_num].transform(data, index, type_data)

        # If we point to another transformer, load the transformer then transform the data
        else:

            if pointer[0] == DEEP_ENTRY_INPUT:
                list_transformers = self.list_input_transformers

            elif pointer[0] == DEEP_ENTRY_LABEL:
                list_transformers = self.list_label_transformers

            elif pointer[0] == DEEP_ENTRY_ADDITIONAL_DATA:
                list_transformers = self.list_additional_data_transformers

            else:
                Notification(DEEP_NOTIF_FATAL, "The following type of transformer does not exist : " + str (pointer[0]))
            transformed_data = list_transformers[pointer[1]].transform(data, index, type_data)

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
        return [self.__create_transformer(config_entry=entry) for entry in entries if entry is not None]

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

        # If the config source is a pointer to another transformer
        if self.__is_pointer(config_entry) is True:
            transformer = Pointer(config_entry)     # Generic Transformer as a pointer
        # If the user wants to create a transformer from scratch
        else:
            config = Namespace(config_entry)
            if not hasattr(config, 'method'):
                Notification(DEEP_NOTIF_FATAL, "The following transformer does not have any method specified : " + str(config_entry))
            # Get the config method in lowercases
            config.method = config.method.lower()
            # If sequential method selected
            if config.method == "sequential":
                transformer = Sequential(**config.get())
            # If someOf method selected
            elif config.method == "someof":
                transformer = SomeOf(**config.get())
            # If oneof method selected
            elif config.method == "oneof":
                transformer = OneOf(**config.get())
            # If the method does not exist
            else:
                Notification(DEEP_NOTIF_FATAL, "The following transformation method does not exist : " + str(config.method))
        return transformer

    @staticmethod
    def __is_pointer(source_path : str) -> bool:
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


