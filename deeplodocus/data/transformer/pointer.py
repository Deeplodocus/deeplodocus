# Python imports
from typing import Union

# Deeplodocus imports
from deeplodocus.utils.notification import Notification
from deeplodocus.utils.flags.entry import *
from deeplodocus.utils.flags.notif import DEEP_NOTIF_INFO, DEEP_NOTIF_FATAL


class Pointer(object):

    def __init__(self, pointer):

        self.name = str(pointer)
        self.pointer_to_transformer = self.__generate_pointer(pointer)

        # TODO : Replace the pointer_to_transformer by a transform entry and a transformer index
        self.transform_entry = None
        self.transformer_index = 0
        #Transformer.__init__(self, config)

    def summary(self):

        Notification(DEEP_NOTIF_INFO, "------------------------------------")
        Notification(DEEP_NOTIF_INFO, "Transformer '" + str(self.name) + "' summary :")
        Notification(DEEP_NOTIF_INFO, "Points to : " + str(self.pointer_to_transformer))
        Notification(DEEP_NOTIF_INFO, "------------------------------------")

    def get_pointer(self):
        """
        AUTHORS:
        --------

        author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the pointer to the other transformer

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: pointer_to_transformer attribute
        """
        return self.pointer_to_transformer

    def __generate_pointer(self, pointer):
        """
        Authors : Alix Leroy,
        Check if the transformer has to point to another transformer
        :return: The pointer to another transformer
        """
        if str(pointer)[0] == "*":                 # If we have defined the transformer as a pointer to another transformer
            path_splitted = str(pointer[1:]).split(":") # Get the path to the other transformer (Entrytype:Number)

            if len(path_splitted) != 2 :
                Notification(DEEP_NOTIF_FATAL, "The following transformer does not point correctly to another transformer (path:number format required), please check the documentation : " + str(pointer))

            # Check that the pointer type is valid and convert it to an integer for efficiency during training or testing
            path_splitted[0] = self.__convert_pointer_type(path_splitted[0])

            try:
                int(path_splitted[1])
            except:
                #if isinstance(path_splitted[1], int) is False:
                Notification(DEEP_NOTIF_FATAL, "The second argument of the following transformer's pointer is not an integer : " + str(pointer),)

            return [path_splitted[0], int(path_splitted[1])] # Return type and index of the pointer

        else:
            Notification(DEEP_NOTIF_FATAL, "The following transformer is not a pointer : " + str(pointer))

    def __convert_pointer_type(self, pointer_type : Union[str, int, Flag]):
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :param pointer_type:
        :return:
        """


        type = str(pointer_type).lower()

        if type == "inputs" or type == "input":
            return DEEP_ENTRY_INPUT

        elif  type == "labels" or type == "label":
            return DEEP_ENTRY_LABEL

        elif type == "additional_data":
            return DEEP_ENTRY_ADDITIONAL_DATA

        else :
            Notification(DEEP_NOTIF_FATAL, "The type of the following transformer's pointer does not exist, please check the documentation : " + str(self.name))

