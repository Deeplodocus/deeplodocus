from deeplodocus.utils.singleton import Singleton


class FlagIndexer(metaclass=Singleton):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Indexer class.
    Manages the unique indexes for the flags
    Inherits from Singleton : Only one unique instance of the class exists while running.
    Can be called anywhere in Deeplodocus
    """

    def __init__(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize the Indexer
        The first index is zero
        Each new index is incremented
        """
        self.index = -1  # Initialize at -1 so that the first index is 0 after increment

    def generate_unique_index(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Create a unique index for a flag and increment for the next one

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return self.index (int): The unique index
        """

        self.index += 1
        return self.index
