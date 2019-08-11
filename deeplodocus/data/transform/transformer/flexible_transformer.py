from typing import Union
from typing import List

# Deeplodocus imports
from deeplodocus.utils.notification import Notification
from deeplodocus.data.transform.transformer.transformer import Transformer
from deeplodocus.utils.namespace import Namespace

# Deeplodocus flags
from deeplodocus.flags import DEEP_NOTIF_INFO


class FlexibleTransformer(Transformer):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Flexible transformer including mandatory transforms before and after random selection of operations
    """

    def __init__(self, name: str, mandatory_transforms_start: Union[Namespace, List[dict]], transforms: Union[Namespace, List[dict]], mandatory_transforms_end: Union[Namespace, List[dict]]):
        """
        AUTHORS:
        """

        super().__init__(name, transforms)

        self.list_mandatory_transforms_start = self.__fill_transform_list(mandatory_transforms_start)
        self.list_mandatory_transforms_end = self.__fill_transform_list(mandatory_transforms_end)


    def summary(self):
        """
        AUTHORS:
        --------

        author: Alix Leroy

        DESCRIPTION:
        -----------

        Print the summary of the tranformer

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """

        Notification(DEEP_NOTIF_INFO, "Transformer '" + str(self.__name) + "' summary :")

        # MANDATORY TRANSFORMS START
        if len(self.list_mandatory_transforms_start) > 0:
            Notification(DEEP_NOTIF_INFO, " Mandatory transforms at start:")
            for t in self.list_mandatory_transforms_start:
                Notification(DEEP_NOTIF_INFO, "--> Name : " + str(t["name"]) + " , Args : " + str(t["kwargs"]) + ", Module path: " + str(t["module_path"]))

        # TRANSFORMS
        if len(self.list_transforms) > 0:
            Notification(DEEP_NOTIF_INFO, " Transforms :")
            for t in self.list_transforms:
                Notification(DEEP_NOTIF_INFO, "--> Name : " + str(t["name"]) + " , Args : " + str(t["kwargs"]) + ", Module path: " + str(t["module_path"]))


        # MANDATORY TRANSFORMS END
        if len(self.list_mandatory_transforms_end) > 0:
            Notification(DEEP_NOTIF_INFO, " Mandatory transforms at end:")
            for t in self.list_mandatory_transforms_end:
                Notification(DEEP_NOTIF_INFO, "--> Name : " + str(t["name"]) + " , Args : " + str(t["kwargs"]) + ", Module path: " + str(t["module_path"]))