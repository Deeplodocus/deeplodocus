import time

# Import transformers
from deeplodocus.data.transformer.transformer import Transformer
from deeplodocus.data.transformer.one_of import OneOf
from deeplodocus.data.transformer.sequential import Sequential
from deeplodocus.data.transformer.some_of import SomeOf

from deeplodocus.utils.notification import Notification
from deeplodocus.utils.namespace import Namespace
from deeplodocus.utils.types import *



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



    def __init__(self, parameters, write_logs=True)->None:
        """
        AUTHORS:
        --------

        author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize the TransformManager object

        PARAMETERS:
        -----------

        :param transform_parameters: The list of parameters of the

        RETURN:
        -------

        None
        """

        self.write_logs = write_logs

        self.parameters = parameters

        # Handle name
        if hasattr(parameters, 'name'):
            self.name = str(parameters.name)
        else:
            self.name = "No name given - " + str(time.time())

        # Handle inputs
        if hasattr(parameters, 'inputs') or hasattr(parameters, 'input'):
            self.list_input_transformers = self.__load_transformers(parameters.inputs)
        else:
            self.list_input_transformers = []

        # Handle labels
        if hasattr(parameters, 'labels') or hasattr(parameters, 'label'):
            self.list_label_transformers = self.__load_transformers(parameters.labels)
        else:
            self.list_label_transformers = []

        # Handle additional_data
        if hasattr(parameters, 'additional_data'):
            self.list_additional_data_transformers = self.__load_transformers(parameters.additional_data)
        else:
            self.list_additional_data_transformers = []

        # Print summary of the transformer
        self.__summary()

    def update(self, parameters) -> None:
        """
        AUTHORS:
        --------

        author: Alix Leroy

        DESCRIPTION:
        ------------

        Update all the parameters of the TransformerManager

        PARAMETERS:
        -----------

        :param parameters: The list of parameters to update

        RETURN:
        -------

        :return: None
        """

        try:
            self.parameters = parameters

            if hasattr(parameters, 'name'):
                self.name = str(parameters.name)

            # Handle inputs
            if hasattr(parameters, 'inputs') or hasattr(parameters, 'input'):
                self.list_input_transformers = self.__load_transformers(parameters.inputs)
            else:
                self.list_input_transformers = []

            # Handle labels
            if hasattr(parameters, 'labels') or hasattr(parameters, 'label'):
                self.list_label_transformers = self.__load_transformers(parameters.labels)
            else:
                self.list_label_transformers = []

            # Handle additional_data
            if hasattr(parameters, 'additional_data'):
                self.list_additional_data_transformers = self.__load_transformers(parameters.additional_data)
            else:
                self.list_additional_data_transformers = []

            Notification(DEEP_SUCCESS, "The TransformManager '" + str(self.name) +"' has succesfully been updated.", write_logs=self.write_logs)

        except:
            Notification(DEEP_ERROR,
                         "An error occurred while updating the TransformManager '" + str(self.name) +"'. Please check the given configuration",
                         write_logs=self.write_logs)


    def transform(self, data, index, type_data, entry_type , entry_num):
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
        if entry_type == DEEP_TYPE_INPUT:
            list_transformers = self.list_input_transformers
        elif entry_type == DEEP_TYPE_LABEL:
            list_transformers = self.list_label_transformers
        elif entry_type == DEEP_TYPE_ADDITIONAL_DATA:
            list_transformers = self.list_additional_data_transformers
        else:
            Notification(DEEP_FATAL, "The following type of transformer does not exist : " + str (entry_type))


        # Check if the transformer points to another transformer
        pointer = list_transformers[entry_num].get_pointer()

        # If we do not point to another transformer, transform directly the data
        if pointer is None:
        # Transform
            transformed_data = list_transformers[entry_num].transform(data, index, type_data)

        # If we point to another transformer, load the transformer then transform the data
        else:

            if pointer[0] == DEEP_TYPE_INPUT:
                list_transformers = self.list_input_transformers

            elif pointer[0] == DEEP_TYPE_LABEL:
                list_transformers = self.list_label_transformers

            elif pointer[0] == DEEP_TYPE_ADDITIONAL_DATA:
                list_transformers = self.list_additional_data_transformers

            else:
                Notification(DEEP_FATAL, "The following type of transformer does not exist : " + str (pointer[0]))

            transformed_data = list_transformers[pointer[1]].transform(data, index, type_data)

        return transformed_data



    def __summary(self):
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
            transformer.summary()

        # Labels
        for transformer in self.list_label_transformers:
            transformer.summary()

        # Additional data
        for transformer in self.list_additional_data_transformers:
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

        transformers_list = []

        # If there is only one entry not in a list format (input, label, additional_data)
        if entries is not list:
            entries = [entries]



        # Load and create the transformers and then add them to the transformers list
        for entry in entries:
            # Check if the entry is None
            if entry is None or entry == "":
                transformers_list.append(None)
            else:
                transformer = self.__create_transformer(config_entry=entry)
                transformers_list.append(transformer)

        # return the list of transformers
        return transformers_list


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
            transformer = Transformer(config_entry) # Generic Transformer

        # If the user wants to create a transformer from scratch
        else:

            config = Namespace(yaml_path=config_entry[0])

            if hasattr(config, 'method') is False:
                Notification(DEEP_FATAL, "The following transformer does not have any method specified : " + str(config_entry), write_logs=self.write_logs)

            # Get the config method in lowercases
            config.method = config.method.lower()

            # If sequential method selected
            if config.method == "sequential":
                transformer = Sequential(config)

            # If someOf method selected
            elif config.method == "someof":
                transformer = SomeOf(config)

            # If oneof method selected
            elif config.method == "oneof":
                transformer = OneOf(config)

            # If the method does not exist
            else:
                Notification(DEEP_FATAL , "The following transformation method does not exist : " + str(config.method))

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


