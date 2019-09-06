# Python imports
from typing import Any
from typing import Tuple
from typing import Optional
from typing import Union

# Deeplodocus imports
from deeplodocus.utils.notification import Notification

# Deeplodocus flags
from deeplodocus.flags import DEEP_NOTIF_ERROR, DEEP_NOTIF_WARNING


class Source(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Source class
    A generic Source class for loading data
    """

    def __init__(self,
                 index: int,
                 is_loaded: bool = False,
                 is_transformed: bool = False,
                 num_instances: Optional[int] = None,
                 instance_id: int = 0
                 ):

        # ID of the source in the DataStore
        self.index = index

        # Whether the gotten data is loaded
        self.is_loaded = is_loaded

        # Whether the gotten data is transformed or has to go through the transformation process
        self.is_transformed = is_transformed

        # Number of instances in the source
        # None if not defined yet or if unlimited data (e.g. agent)
        self.num_instances = num_instances

        # Id of the instance to load
        self.instance_id = instance_id

        # Whether the source has been checked correctly
        self.checked = False



    def __getitem__(self, index: int) -> Tuple[Any, bool, bool]:
        pass

    def __len__(self) -> Optional[int]:
        return self.num_instances

    def check(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check the source is correctly loaded

        :return:
        """
        if self.num_instances is None:
            self.num_instances = self.compute_length()

        # Consider the Source as checked
        self.checked = True

    def verify_custom_source(self) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Verify the Source has been correctly checked
        Useful when the Source is a custom one and the user forgot to use super().check() in its custom check() function

        PARAMETERS:
        -----------

        :param: None

        RETURN:
        -------

        :return: None
        """
        if self.checked is False:
            Notification(DEEP_NOTIF_ERROR, "The source with the ID %i is not checked properly, please make sure you used super().check() in the custom check() function of your Source")

    def compute_length(self) -> Optional[int]:
        Notification(DEEP_NOTIF_WARNING, "You forgot to compute the length of a Source instance")
        return None

    ###########
    # GETTERS #
    ###########


    def get_index(self):
        return self.index

    def get_instance_id(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get self.instance_id

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return self.instance_id(int): Index of the desired instance in the cache
        """
        return self.instance_id

    def get_num_instances(self) -> Union[int, None]:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get self.num_instances

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return self.num_instances(Union[int, None]): Number of instances in the Source (None is unlimited)
        """
        return self.num_instances

    def get_checked(self) -> bool:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get self.checked

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return self.checked(bool): Whether the Source instance has already been check or not
        """
        return self.checked
