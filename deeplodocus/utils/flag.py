# Python imports
from typing import List

# Deeplodocus imports
from deeplodocus.utils.flag_indexer import FlagIndexer


class Flag(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Complex Flag class
    Create a flag with a full description, an auto-generated index and a list of accepted names.
    """

    def __init__(self, description: str, names : List[str]):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize a new flag.
        Each new flag contains a unique index and a corresponding name.
        The description is optional and not recommended for memory efficiency

        PARAMETERS:
        -----------

        :param index (int): The index of the flag
        :param names (List[str]): Names of the flag
        :param description (str, Optional): Description of the flag

        RETURN:
        -------

        None
        """
        self.index = FlagIndexer().generate_unique_index()
        self.names = names
        self.description = description

    def __call__(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Call the flag to get the index

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return self.index (int): The index of the flag
        """
        return self.index

    def __str__(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Print the description of the flag with the corresponding index

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return (str): The Description of the flag with the corresponding index
        """
        return "{0} : (id : {1})".format(self.description, self.index)


    def corresponds(self, name : str):
        """
        AUTHORS:
        --------

        :author: Alix Leroy


        DESCRIPTION:
        ------------

        Check if a name is part of the

        PARAMETERS:
        -----------

        :param name (str): The name to check

        RETURN:
        -------

        :return (bool): Whether the string is part of the names or not
        """

        if name in self.names:
            return True
        else:
            return False

    """
    "
    " GETTERS
    "
    """
    def get_index(self):
        return self.index

    def get_names(self):
        return self.names

    def get_description(self):
        return self.description

