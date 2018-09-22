
# Import transformers
from deeplodocus.data.transformer.one_of import OneOf
from deeplodocus.data.transformer.sequential import Sequential
from deeplodocus.data.transformer.some_of import SomeOf
from deeplodocus.utils.notification import Notification
from deeplodocus.utils.namespace import Namespace
from deeplodocus.utils.transformer.transformer import Transformer
from deeplodocus.utils.notification import DEEP_CRITICAL

class Transform(object):



    def __init__(self, transform_parameters = None):
        """
        Author: Alix Leroy
        Offline or online transformation of the data
        """


        self.list_input_transformers = self.__load_transformers(transform_parameters.inputs)
        self.list_label_transformers = self.__load_transformers(transform_parameters.labels)
        self.list_additional_data_transformers = self.__load_transformers(transform_parameters.additional_data)

        # Print summary of the transformer
        self.__summary()


    def transform(self, data, index, type_data, entry_type , entry_num):
        """
        Authors : Alix Leroy
        :param data: data to transform
        :param index: The index of the instance in the Data Frame
        :param type_data: type of data to transform
        :param entry_type: type of entry (input, label, additional_data)
        :param entry_num: num of the entry (input1, input2, ...)
        :return: transformed object
        """

        list_transformers = None


        if entry_type == "inputs":
            list_transformers = self.list_input_transformers
        elif entry_type == "labels":
            list_transformers = self.list_label_transformers
        elif entry_type == "additional_data":
            list_transformers = self.list_additional_data_transformers
        else:
            Notification(DEEP_FATAL, "The following type of transformer does not exist : " + str (entry_type))


        # Check if the transformer points to another transformer
        pointer = list_transformers[entry_num].get_pointer()

        if pointer is None:
        # Transform
            transformed_data = list_transformers[entry_num].transform(data, index, type_data)

        else:
            if pointer[0] == "inputs":
                list_transformers = self.list_input_transformers
            elif pointer[0] == "labels":
                list_transformers = self.list_label_transformers
            elif pointer[0] == "additional_data":
                list_transformers = self.list_additional_data_transformers
            else:
                Notification(DEEP_FATAL, "The following type of transformer does not exist : " + str (pointer[0]))

            transformed_data = list_transformers[pointer[1]].transform(data, index, type_data)

        return transformed_data



    def __summary(self):
        """
        Authors : Alix Leroy,
        Display the summary of the Transformers
        :return: None
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
        Authors : ALix Leroy
        Load the transformers
        :param: entries : a list of paths for transformer config files
        :return: A list of transformers
        """

        transformers = []

        # If there is only one entry not in a list format (input, label, additional_data)
        if entries is not list:
            entries = [entries]

        # Load and create the transformers and then add them to the transformers list
        for entry in entries:

            transformer_config = Namespace(entry)
            transformer = self.__create_transformer(config=transformer_config)
            transformers.append(transformer)

        # return the list of transformers
        return transformers


    def __create_transformer(self, config, pointer=False):
        """
        Author : Alix Leroy,
        Create the appropriate transformer
        :return: A transformer
        """

        # Get the config method in lowercases
        config.method = config.method.lower()

        if pointer is True:
            transformer = Transformer(config) # Generic Transformer

        # If sequential method selected
        elif config.method == "sequential":
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


