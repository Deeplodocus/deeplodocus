from deeplodocus.utils.notification import Notification
from deeplodocus.utils.types import *
from deeplodocus.data.transformer.transformer import Transformer


class Pointer(object):

    def __init__(self, pointer, write_logs=False):

        self.write_logs = write_logs
        self.name = str(pointer)
        self.pointer_to_transformer = self.__generate_pointer(pointer)

        #Transformer.__init__(self, config)

    def summary(self):

        Notification(DEEP_INFO, "------------------------------------", write_logs=self.write_logs)
        Notification(DEEP_INFO, "Transformer '" + str(self.name) + "' summary :", write_logs=self.write_logs)
        Notification(DEEP_INFO, "Points to : " + str(self.pointer_to_transformer), write_logs=self.write_logs)
        Notification(DEEP_INFO, "------------------------------------", write_logs=self.write_logs)

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
                Notification(DEEP_FATAL, "The following transformer does not point correctly to another transformer (path:number format required), please check the documentation : " + str(pointer), write_logs=self.write_logs)

            # Check that the pointer type is valid and convert it to an integer for efficiency during training or testing
            path_splitted[0] = self.__convert_pointer_type(path_splitted[0])

            try:
                int(path_splitted[1])
            except:
                #if isinstance(path_splitted[1], int) is False:
                Notification(DEEP_FATAL, "The second argument of the following transformer's pointer is not an integer : " + str(pointer), write_logs=self.write_logs)

            return [path_splitted[0], int(path_splitted[1])] # Return type and index of the pointer

        else:
            Notification(DEEP_FATAL, "The following transformer is not a pointer : " + str(pointer), write_logs=self.write_logs)

    def __convert_pointer_type(self, pointer_type):

        type = str(pointer_type).lower()

        if type == "inputs" or type == "input":
            return DEEP_TYPE_INPUT

        elif  type == "labels" or type == "label":
            return DEEP_TYPE_LABEL

        elif type == "additional_data":
            return DEEP_TYPE_ADDITIONAL_DATA

        else :
            Notification(DEEP_FATAL, "The type of the following transformer's pointer does not exist, please check the documentation : " + str(self.name), write_logs=self.write_logs)

