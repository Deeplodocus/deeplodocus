# Python imports
from typing import Union
from typing import Tuple

# Deeplodocus imports
from deeplodocus.utils.notification import Notification
from deeplodocus.utils.generic_utils import get_corresponding_flag

# Deeplodocus flags
from deeplodocus.utils.flags.entry import *
from deeplodocus.utils.flags.notif import DEEP_NOTIF_INFO, DEEP_NOTIF_FATAL
from deeplodocus.utils.flags.flag_lists import DEEP_LIST_POINTER_ENTRY


class Pointer(object):

    def __init__(self, pointer):

        self.name = str(pointer)
        self.transformer_entry, self.transformer_index = self.__generate_pointer(pointer)
        #Transformer.__init__(self, config)

    def summary(self) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Print the summary of the Pointer

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        Notification(DEEP_NOTIF_INFO, "------------------------------------")
        Notification(DEEP_NOTIF_INFO, "Transformer '" + str(self.name) + "' summary :")
        Notification(DEEP_NOTIF_INFO, "Points to the  %ith %s." %(self.transformer_index, self.transformer_entry.get_name()))
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
        return self.transformer_entry, self.transformer_index

    def __generate_pointer(self, pointer : str) -> Tuple[Flag, int]:
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
            entry_type = self.__convert_pointer_type(path_splitted[0])

            try:
                entry_index = int(path_splitted[1])
            except:
                Notification(DEEP_NOTIF_FATAL, "The second argument of the following transformer's pointer is not an integer : " + str(pointer),)

            return entry_type, entry_index # Return type and index of the pointer

        else:
            Notification(DEEP_NOTIF_FATAL, "The following transformer is not a pointer : " + str(pointer))

    def __convert_pointer_type(self, pointer_type : Union[str, int, Flag]) -> Flag:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Convert the pointer type to the actual entry flag

        PARAMETERS:
        -----------

        :param pointer_type (str): The pointer type the user wants

        RETURN:
        -------

        :return flag(Flag): The corresponding flag of entry type
        """

        flag = get_corresponding_flag(flag_list=DEEP_LIST_POINTER_ENTRY,
                                      info = pointer_type,
                                      fatal=True)
        if flag is None:
            Notification(DEEP_NOTIF_FATAL,
                         "The type of the following transformer's pointer does not exist :' %s'. "
                         "Please check the documentation." % str(self.name))
        else:
            return flag

    @staticmethod
    def has_transforms():
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Whether the transformer contains transforms

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return (bool): Whether the transformer contains transforms
        """
        return True
