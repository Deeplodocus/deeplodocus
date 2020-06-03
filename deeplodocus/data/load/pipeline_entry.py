# Python imports
from typing import Optional
from typing import List
from typing import Any
import weakref

# Deeplodocus imports
from deeplodocus.data.load.formatter import Formatter
from deeplodocus.utils.flag import Flag



class PipelineEntry(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    A PipelineEntry class.
    Is the entry point of the model, losses and metrics
    Format the data
    """

    def __init__(self,
                 index: int,
                 dataset: weakref,
                 entry_type: Flag,
                 entry_type_index: int,
                 convert_to: Optional[List[int]] = None,
                 move_axis: Optional[List[int]] = None):

        # Index of the PipelineEntry instance
        self.index = index

        # Entry type (Input, Label, Additional Data)
        self.entry_type = entry_type

        # Index of the entry type in the Dataset
        self.entry_type_index = entry_type_index

        # Dataset
        self.dataset = dataset

        # Data formatter
        self.formatter = Formatter(
            pipeline_entry=weakref.ref(self),
            convert_to=convert_to,
            move_axis=move_axis
        )

    def format(self, data: Any) -> Any:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Call the Formatter instance to format the data
            1) Format the data type
            2) Move the axis to a define sequence

        PARAMETERS:
        -----------

        :param data (Any): The data to format

        RETURN:
        -------

        :return (Any): The formatted data
        """
        return self.formatter.format(data=data, entry_type=self.entry_type)

    ###########
    # GETTERS #
    ###########

    def get_index(self) -> int:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the PipelineEntry index

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return self.index(int): The PipelineEntry index
        """
        return self.index

    def get_dataset(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the weakref of the Dataset whose the Pipeline belongs to

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return self.index(int): The weakref of the Dataset
        """
        return self.dataset()

    def get_entry_type(self) -> Flag:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the entry type

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return self.entry_type(Flag): The entry type
        """
        return self.entry_type

    def get_entry_type_index(self) -> int:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the PipelineEntry index for the type it belong (input, label, additional_data)

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return self.entry_type_index(int): The PipelineEntry index
        """
        return self.entry_type_index
