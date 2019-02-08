# Python imports
from typing import List
from typing import Union

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

    def __init__(self, name: str, description: str, names: List[str]):
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

        :param name(str): Official name of the flag
        :param names (List[str]): Names of the flag
        :param description (str): Description of the flag

        RETURN:
        -------

        None
        """
        self.name = name
        self.names = names
        self.description = description
        self.index = FlagIndexer().generate_unique_index()

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
        return "Flag {0} : (id : {1})".format(self.description, self.index)

    def corresponds(self, info : Union[str, int, "Flag", None, bool]) -> bool:
        """
        AUTHORS:
        --------

        :author: Alix Leroy


        DESCRIPTION:
        ------------

        Check if a name is part of the

        PARAMETERS:
        -----------

        :param name (Union[str, int, Flag, None]): The info to check

        RETURN:
        -------

        :return (bool): Whether the info corresponds to the Flag
        """
        # NONE
        if info is None or False:
            return False

        # NAME COMPARISON
        if isinstance(info, str):
            if info.lower() in self.names:
                return True
            else:
                return False

        # INDEX COMPARISON
        elif isinstance(info, int):
            if info == self.index:
                return True
            else:
                return False

        # COMPLETE FLAG COMPARISON
        elif isinstance(info, Flag):
            if info.get_index() == self.index:
                return True
            else:
                return False

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

    def get_name(self):
        return self.name

