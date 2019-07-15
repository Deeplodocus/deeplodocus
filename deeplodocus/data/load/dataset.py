

# Python import
import numpy as np
from typing import List
from typing import Union
from typing import Any
import random
import weakref

# Deeplodocus imports
from deeplodocus.data.load.entry import Entry
from deeplodocus.utils.generic_utils import get_corresponding_flag

# Deeplodocus flags
from deeplodocus.utils.flags.flag_lists import *
from deeplodocus.utils.flags.msg import *
from deeplodocus.utils.flags.notif import *
from deeplodocus.utils.flags.shuffle import *
from deeplodocus.utils.notification import Notification
from deeplodocus.utils.namespace import Namespace


class Dataset(object):
    """
    AUTHORS:
    --------

    :author : Alix Leroy
    :author: Samuel Westlake


    DESCRIPTION:
    ------------

    A Dataset class to manage the data given by the config files.
    The following class permits :
        - Data checking
        - Smart data loading
        - Data formatting
        - Data transform (through the TransformManager class)


    The dataset is split into 3 subsets :
        - Inputs : Data given as input to the network
        - Labels : Data given as output (ground truth) to the network (optional)
        - Additional data : Data given to the loss function (optional)

    The dataset class supports 2 different image processing libraries :
        - PILLOW (fork of PIL) as default
        - OpenCV (usage recommended for efficiency)
    """

    def __init__(self,
                 inputs=None,
                 labels=None,
                 additional_data=None,
                 number=None,
                 name="Default",
                 use_raw_data=True,
                 cv_library: Flag = DEEP_LIB_PIL,
                 transform_manager=None):
        """
        AUTHORS:
        --------

        author: Alix Leroy
        author:

        DESCRIPTION:
        ---------

        Initialize the dataset

        PARAMETERS:
        -----------

        :param inputs: A list of input files/folders/list of files/folders
        :param labels: A list of label files/folders/list of files/folders
        :param additional_data: A list of additional data files/folders/list of files/folders
        :param use_raw_data: Boolean : Whether to feed the network with raw data or always apply transforms on it
        :param transform_manager: A TransformManager instance
        :param name: Name of the dataset

        RETURN:
        -------

        :return: None
        """
        self.name = name
        self.list_inputs = self.__generate_entries(entries=self.__check_null_entry(inputs),
                                                   entry_type=DEEP_ENTRY_INPUT)
        self.list_labels = self.__generate_entries(entries=self.__check_null_entry(labels),
                                                   entry_type=DEEP_ENTRY_LABEL)
        self.list_additional_data = self.__generate_entries(entries=self.__check_null_entry(additional_data),
                                                            entry_type=DEEP_ENTRY_ADDITIONAL_DATA)
        self.number_raw_instances = self.__calculate_number_raw_instances()
        self.length = self.__compute_length(desired_length=number,
                                            num_raw_instances=self.number_raw_instances)
        self.transform_manager = transform_manager
        self.use_raw_data = use_raw_data
        self.warning_video = None
        self.cv_library = None
        self.set_cv_library(cv_library)
        self.item_order = np.arange(self.length)

    def __getitem__(self, index: int):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the selected item
        The item is selected accordingly to the required function

        PARAMETERS:
        -----------

        :param index:

        RETURN:
        -------

        :return instance: Loaded and possibly transformed instance to be given to the training
        """

        # If the index given is too big => Error
        if index >= self.length:
            Notification(DEEP_NOTIF_FATAL, "The given instance index is too high : " + str(index))
        # Else we get the random generated index
        else:
            index = self.item_order[index]

        # If we ask for a not existing index we use the modulo and consider the data to have to be augmented
        if index >= self.number_raw_instances:
            augment = True
        # If we ask for a raw data, augment it only if required by the user
        else:
            augment = not self.use_raw_data

        # Extract lists of raw data for the selected index
        labels = []
        additional_data = []
        if not self.list_labels:
            if not self.list_additional_data:
                inputs = self.__load(entries=self.list_inputs,
                                     index=index,
                                     augment=augment)
            else:
                inputs = self.__load(entries=self.list_inputs,
                                     index=index,
                                     augment=augment)
                additional_data = self.__load(entries=self.list_additional_data,
                                              index=index,
                                              augment=augment)
        else:
            if not self.list_additional_data:
                inputs = self.__load(entries=self.list_inputs,
                                     index=index,
                                     augment=augment)
                labels = self.__load(entries=self.list_labels,
                                     index=index,
                                     augment=augment)
            else:
                inputs = self.__load(entries=self.list_inputs,
                                     index=index,
                                     augment=augment)
                labels = self.__load(entries=self.list_labels,
                                     index=index,
                                     augment=augment)
                additional_data = self.__load(entries=self.list_additional_data,
                                              index=index,
                                              augment=augment)
        return inputs, labels, additional_data

    """
    "
    " LENGTH
    "
    """
    def __len__(self) -> int:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the length of the data set

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return (int): The length of the data set
        """

        return self.length

    @staticmethod
    def __compute_length(desired_length: int, num_raw_instances: int) -> int:
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Calculate the length of the dataset

        PARAMETERS:
        -----------

        :param desired_length(int): The desired number of instances
        :param num_raw_instances(int): The actual number of instance in the sources

        RETURN:
        -------

        :return (int): The length of the dataset
        """

        if desired_length is None:
            Notification(DEEP_NOTIF_INFO, DEEP_MSG_DATA_NO_LENGTH % num_raw_instances)
            return num_raw_instances
        else:
            if desired_length > num_raw_instances:
                Notification(DEEP_NOTIF_INFO, DEEP_MSG_DATA_GREATER % (desired_length, num_raw_instances))
                return desired_length
            elif desired_length < num_raw_instances:
                Notification(DEEP_NOTIF_WARNING, DEEP_MSG_DATA_SHORTER % (num_raw_instances, desired_length))
                return desired_length
            else:
                Notification(DEEP_NOTIF_INFO, DEEP_MSG_DATA_LENGTH % num_raw_instances)
                return desired_length

    def __calculate_number_raw_instances(self) -> int:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Calculate the theoretical number of instances in each epoch
        The first given file/folder stands as the frame to count

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return num_raw_instances (int): theoretical number of instances in each epoch
        """
        # Calculate for the first entry
        try:
            num_raw_instances = self.list_inputs[0].__len__()
        except IndexError as e:
            Notification(
                DEEP_NOTIF_FATAL,
                "IndexError : %s : %s" % (str(e), DEEP_MSG_DATA_INDEX_ERROR % self.name),
                solutions=[
                    DEEP_MSG_DATA_INDEX_ERROR_SOLUTION_1 % self.name,
                    DEEP_MSG_DATA_INDEX_ERROR_SOLUTION_2 % self.name
                ]
            )

        # Gather all the entries in one list
        entries = self.list_inputs + self.list_labels + self.list_additional_data

        # For each entry check if the number of raw instances is the same as the first input
        for index, entry in enumerate(entries):
            n = entry.__len__()

            if n != num_raw_instances:
                Notification(DEEP_NOTIF_FATAL, "Number of instances in " + str(self.list_inputs[0].get_entry_type()) +
                             "-" + str(self.list_inputs[0].get_entry_index()) + " and " + str(entry.get_entry_type()) +
                             "-" + str(entry.get_entry_index()) + " do not match.")
        return num_raw_instances

    def __load(self, entries: List[Entry], index: int, augment: bool) -> List[Any]:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Load one instance of the dataset into memory

        PARAMETERS:
        -----------

        :param entries(List[Entry]): The list of entries to load the instance from
        :param index(int): The index of the instance
        :param augment(bool): Whether to augment the data or not

        RETURN:
        -------

        :return data(List[Any]): The loaded and transformed data
        """
        data = []

        # Get the index of the original instance (before transformation)
        index_raw_instance = index % self.number_raw_instances

        # Gather the item of each entry
        for entry in entries:

            entry_data, is_loaded, is_transformed = entry.__getitem__(index=index_raw_instance)

            # LOAD THE ITEM
            if is_loaded is False:
                entry_data = self.__load_data_from_str(data=entry_data,
                                                       entry=entry)

            # TRANSFORM THE ITEM
            if is_transformed is False:
                entry_data = self.__transform_data(data=entry_data,
                                                   entry=entry,
                                                   index=index,
                                                   augment=augment)

            entry_data = self.__format_data(entry_data, entry)
            data.append(entry_data)

        # If the entry is an input and is single element list we return it as a list so it can be correctly unpacked in the trainer
        if DEEP_ENTRY_INPUT.corresponds(info=entries[0].get_entry_type()) and len(data) == 1:
            return [data]
        else:
            return data

    def __format_data(self, data_entry, entry):
        """
        AUTHORS:
        --------

        :author: Samuel Westlake
        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Method to format data to pytorch conventions.
        Images are converted from (w, h, ch) to (ch, h, w)
        Videos are ...

        PARAMETERS:
        -----------

        :param data_entry: the data instance
        :param entry: the data entry flag

        RETURN:
        -------

        :return: the formatted data entry
        """

        # If we have to format a list of items
        if isinstance(data_entry, list):
            formated_data = []

            for d in data_entry:

                fd = self.__format_data(d, entry)
                formated_data.append(fd)
            data_entry = formated_data


        # Else it is a unique item
        else:

            # Change the format
            data_entry = data_entry.astype(entry.load_as.names[0])

            # Move the axes
            if entry.move_axes is not None:
                data_entry = self.__move_axes(data_entry, entry.move_axes)

            #
            # KEEP THE FOLLOWING LINES WHILE THE DATA FORMATTING IS BEING ON TEST
            #

            # if DEEP_DTYPE_IMAGE.corresponds(entry.data_type):
            #     # Check if no transform return a grayscale image as a 2D image
            #     if data_entry.ndim <= 2:
            #         data_entry = data_entry[:, :, np.newaxis]
            #
            #     # Make image (ch, h, w)
            #     data_entry = np.swapaxes(data_entry, 0, 2)
            #     data_entry = np.swapaxes(data_entry, 1, 2)
            #
            #     data_entry = data_entry.astype(np.float32)

        return data_entry

    def __move_axes(self, data, move_axes) -> np.array:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Move axes of the data

        PARAMETERS:
        -----------

        :param data (np.array): The data needing a axis swap
        :param move_axes(List[int]): The new axes order

        RETURN:
        -------

        :return data (np.array): The data with teh axes swapped
        """

        return np.transpose(data, move_axes)

    def __generate_entries(self, entries: List[Namespace], entry_type: Flag) -> List[Entry]:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Generate a list of Entry instances

        PARAMETERS:
        -----------

        :param entries (List[Namespace]): The list of raw entries in a Namespace format
        :param entry_type (Flag): The flag of the entry type

        RETURN:
        -------

        :return generated_entries (List(Entry)): The list of Entry instances generated
        """
        # Create a weakref to the dataset
        ref = weakref.ref(self)

        # List of generated entries to an Entry class format
        generated_entries = []

        # For each entry in a Namespace format
        for index, entry in enumerate(entries):

            # Check the completeness of the entry
            entry = self.__check_entry_completeness(entry)
            # Create the Entry instance
            new_entry = Entry(sources=entry.source,
                              join=entry.join,
                              data_type=entry.type,
                              load_method=entry.load_method,
                              entry_index=index,
                              entry_type=entry_type,
                              dataset=ref,
                              load_as=entry.load_as,
                              move_axes = entry.move_axes)
            generated_entries.append(new_entry)

        return generated_entries

    @staticmethod
    def __check_entry_completeness(entry: Namespace) -> Namespace:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check if the dictionary formatted entry is complete.
        If not complete, fill the dictionary with default value

        PARAMETERS:
        -----------

        :param entry (Namespace): The entry to check the completeness

        RETURN:
        -------

        :return entry (Namespace): The completed entry

        RAISE:
        ------

        :raise DeepError: Raised if the path is not given
        """

        # SOURCE
        if entry.check("source", None) is False:
            Notification(DEEP_NOTIF_FATAL,
                         "The source was not specified to the following entry : %s" % str(entry.get()))

        # JOIN PATH
        if entry.check("join", None) is False:
            entry.add({"join": None}, None)

        # LOADING METHOD
        if entry.check("load_method", None) is False:
            entry.add({"load_method": "online"})

        # DATA TYPE
        if entry.check("type", None) is False:
            entry.add({"type": None})

        return entry

    def __load_data_from_str(self, data: Union[str, List[str], Any], entry: Entry) -> Union[Any, List[Any]]:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Load a data from a string format to the actual content
        Loads either one item or a list of items

        PARAMETERS:
        -----------

        :param data(Union[str, List[str]]): The data to transform
        :param entry (Entry): The entry to which the item is attached

        RETURN:
        -------

        :return loaded_data(Union[Any, List[Any]]): The loaded data
        """

        loaded_data = None

        # Get data type index (only use the index for efficiency in the loop)
        data_type = entry.get_data_type()

        # Make sure the data contains something
        if data is not None:

            # SEQUENCE
            if isinstance(data, list):
                # If data is a sequence we use the function in a recursive fashion
                loaded_data = []
                for d in data:
                    ld = self.__load_data_from_str(data=d,
                                                   entry=entry)
                    loaded_data.append(ld)

            # IMAGE
            elif DEEP_DTYPE_IMAGE.corresponds(data_type):
                # Load image
                loaded_data = self.__load_image(data)

            # VIDEO
            elif DEEP_DTYPE_VIDEO.corresponds(data_type):
                loaded_data = self.__load_video(data)

            # INTEGER
            elif DEEP_DTYPE_INTEGER.corresponds(data_type):
                loaded_data = int(data)

            # FLOAT NUMBER
            elif DEEP_DTYPE_FLOAT.corresponds(data_type):
                loaded_data = float(data)

            elif DEEP_DTYPE_STRING.corresponds(data_type):
                loaded_data = str(data)

            # NUMPY ARRAY
            elif DEEP_DTYPE_NP_ARRAY.corresponds(data_type):
                loaded_data = np.load(data)

            # Data type not recognized
            else:

                Notification(DEEP_NOTIF_FATAL,
                             "The following data could not be loaded because its type is not recognized : %s.\n"
                             "Please check the documentation online to see the supported types" % data)
        # If the data is None
        else:
            Notification(DEEP_NOTIF_FATAL, DEEP_MSG_DATA_IS_NONE % data)

        return loaded_data

    def __transform_data(self, data: Union[Any, List[Any]], index: int, entry: Entry, augment: bool) \
            -> Union[Any, List[Any]]:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Transform the data
        Transform either one item or a list of item

        PARAMETERS:
        -----------

        :param data(Union[Any, List[Any]]): The data to transform
        :param index (int): The index of the instance
        :param entry (Entry): The entry to which the item is attached

        RETURN:
        -------

        :return transformed_data(Union[Any, List[Any]]): The transformed data
        """

        # If we want to transform a sequence we use the function recursively
        if isinstance(data, list):
            transformed_data = []

            for d in data:
                td = self.__transform_data(data=d,
                                           index=index,
                                           entry=entry,
                                           augment=augment)
                transformed_data.append(td)

        # If it is only one item to transform
        else:
            transformed_data = self.transform_manager.transform(data=data,
                                                                index=index,
                                                                entry=entry,
                                                                augment=augment)

        return transformed_data

    """
    "
    " DATA LOADERS
    "
    """

    def __load_image(self, image_path: str):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Load the image in the image_path

        PARAMETERS:
        -----------

        :param image_path(str): The path of the image to load

        RETURN:
        -------

        :return: The loaded image
        """
        if self.cv_library() == DEEP_LIB_OPENCV():
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        elif self.cv_library() == DEEP_LIB_PIL():
            image = np.array(Image.open(image_path))
        else:
            # Notify the user of invalid cv library
            image = None
            Notification(DEEP_NOTIF_FATAL, DEEP_MSG_CV_LIBRARY_NOT_IMPLEMENTED % self.cv_library.name)

        # Notify the user that the image failed to load
        if image is None:
            Notification(DEEP_NOTIF_FATAL, DEEP_MSG_DATA_CANNOT_LOAD_IMAGE % (self.cv_library.name, image_path))

        # If image is not gray-scale, convert to rgba, else add extra channel
        if image.ndim > 2:
            image = self.__convert_bgra2rgba(image)
        else:
            image = image[:, :, np.newaxis]

        return image

    @staticmethod
    def __convert_bgra2rgba(image):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Convert BGR(alpha) image to RGB(alpha) image

        PARAMETERS:
        -----------

        :param image: image to convert

        RETURN:
        -------

        :return: a RGB(alpha) image
        """

        # Get the number of channels in the image
        _, _, channels = image.shape

        # Handle BGR and BGR(A) images
        if channels == 3:
            image = image[:, :, (2, 1, 0)]
        elif channels == 4:
            image = image[:, :, (2, 1, 0, 3)]
        return image

    def __load_video(self, video_path: str):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------
        Load a video

        PARAMETERS:
        -----------

        :param video_path->str: absolute path to a video

        RETURN:
        -------

        :return: a list of frame from the video
        """
        self.__throw_warning_video()
        video = []
        # If the computer vision library selected is OpenCV
        if self.cv_library() == DEEP_LIB_OPENCV():
            # try to load the file
            cap = cv2.VideoCapture(video_path)
            while True:
                _, frame = cap.read()
                if frame is None:
                    break
                video.append(self.__convert_bgra2rgba(frame))
            cap.release()
        else:
            Notification(DEEP_NOTIF_FATAL,
                         "The video could not be loaded because OpenCV is not selected as the Computer Vision library")
        return video

    def __throw_warning_video(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Warn the user of the unstable video mode.

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        if self.warning_video is None:
            Notification(DEEP_NOTIF_WARNING, "The video mode is not fully supported. "
                                             "We deeply suggest you to use sequences of images.")
            self.warning_video = 1

    def set_cv_library(self, cv_library: Flag) -> None:
        """
         AUTHORS:
         --------

         :author: Samuel Westlake
         :author: Alix Leroy

         DESCRIPTION:
         ------------

         Set self.cv_library to the given value and import the corresponding cv library

         PARAMETERS:
         -----------

         :param cv_library: (Flag): The flag of the computer vision library selected

         RETURN:
         -------

         None
         """
        self.cv_library = get_corresponding_flag(flag_list=DEEP_LIST_CV_LIB, info=cv_library)
        self.__import_cv_library(cv_library=cv_library)

    @staticmethod
    def __import_cv_library(cv_library : Flag) -> None:
        """
        AUTHORS:
        --------

        :author: Samuel Westlake
        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Imports either cv2 or PIL.Image dependant on the value of cv_library

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        None
        """
        if DEEP_LIB_OPENCV.corresponds(info=cv_library):
            try:
                # Notification(DEEP_NOTIF_INFO, DEEP_MSG_CV_LIBRARY_SET % "OPENCV")
                global cv2
                import cv2
            except ImportError as e:
                Notification(DEEP_NOTIF_ERROR, str(e))
        elif DEEP_LIB_PIL.corresponds(info=cv_library):
            try:
                # Notification(DEEP_NOTIF_INFO, DEEP_MSG_CV_LIBRARY_SET % "PILLOW")
                global Image
                from PIL import Image
            except ImportError as e:
                Notification(DEEP_NOTIF_ERROR, str(e))
        else:
            Notification(DEEP_NOTIF_ERROR, DEEP_MSG_CV_LIBRARY_NOT_IMPLEMENTED % cv_library)

    @staticmethod
    def __check_null_entry(entry):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check if an entry is Null

        PARAMETERS:
        -----------

        :param entry: The entry to check

        RETURN:
        -------

        :return ->list: The formatted entry
        """

        try:
            if entry is None:
                return []
            elif entry[0] is None:
                return []
            else:
                return entry
        except IndexError:
            Notification(DEEP_NOTIF_FATAL, DEEP_MSG_DATA_ENTRY % entry)

    def shuffle(self, method: Flag) -> None:
        """
        AUTHORS:
        --------

        author: Alix Leroy

        DESCRIPTION:
        ------------

        Shuffle the dataframe containing the data

        PARAMETERS:
        -----------

        :param method: (Flag): The shuffling method Flag

        RETURN:
        -------

        :return: None
        """
        # ALL DATASET
        if DEEP_SHUFFLE_ALL.corresponds(info=method):
            self.item_order = np.random.randint(0, high=self.length, size=(self.length,))
            Notification(DEEP_NOTIF_INFO, DEEP_MSG_SHUFFLE_COMPLETE % method.name)

        # NONE
        elif DEEP_SHUFFLE_NONE.corresponds(info=method):
            pass

        # BATCHES
        elif DEEP_SHUFFLE_BATCHES.corresponds(info=method):
            Notification(DEEP_NOTIF_ERROR, "Batch shuffling not implemented yet.")

        # RANDOM PICK
        elif DEEP_SHUFFLE_RANDOM_PICK.corresponds(info=method):
            self.item_order = random.sample(range(0, self.number_raw_instances), self.length)
            Notification(DEEP_NOTIF_INFO, DEEP_MSG_SHUFFLE_COMPLETE % method.name)

        # WRONG FLAG
        else:
            Notification(DEEP_NOTIF_ERROR, DEEP_MSG_SHUFFLE_NOT_FOUND % method.name)

        # Reset the TransformManager
        self.reset()

    def reset(self) -> None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------
        Reset the transform_manager

        PARAMETERS:
        -----------
        None

        RETURN:
        -------
        :return: None
        """
        if self.transform_manager is not None:
            self.transform_manager.reset()

    """
    "
    " GETTERS
    "
    """

    def get_name(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the name of the dataset

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return (str): The name of the dataset
        """
        return self.name

    def get_transform_manager(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the Transform Manager linked to the dataset

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return (TransformManager): The transform manager
        """
        return self.transform_manager
