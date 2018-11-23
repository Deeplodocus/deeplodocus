import pandas as pd
import numpy as np
import mimetypes
import os

from deeplodocus.utils.generic_utils import sorted_nicely
from deeplodocus.utils.generic_utils import get_int_or_float
from deeplodocus.utils.generic_utils import is_np_array
from deeplodocus.utils.notification import Notification
from deeplodocus.utils.errors import error_entry_array_size
from deeplodocus.utils.flags import *


class Dataset(object):
    """
    AUTHORS:
    --------

    author : Alix Leroy


    DESCRIPTION:
    ------------

    A dataset class to manage the data given by the config files.
    The following class permits :
        - Data checking
        - Smart data loading
        - Data formatting
        - Data transform (through the TransformManager class)


    The dataset is splitted into 3 subsets :
        - Inputs : Data given as input to the network
        - Labels : Data given as output (ground truth) to the network (optional)
        - Additional data : Data given to the loss function (optional)

    The dataset class supports 2 different image processing libraries :
        - PILLOW (fork of PIL) as default
        - OpenCV (usage recommended for efficiency)
    """

    def __init__(self, list_inputs, list_labels, list_additional_data,
                 use_raw_data=True,
                 transform_manager=None,
                 cv_library=DEEP_LIB_OPENCV,
                 name="Default"):
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

        :param list_inputs: A list of input files/folders/list of files/folders
        :param list_labels: A list of label files/folders/list of files/folders
        :param list_additional_data: A list of additional data files/folders/list of files/folders
        :param use_raw_data: Boolean : Whether to feed the network with raw data or always apply transforms on it
        :param transform: A transform object
        :param cv_library: The computer vision library to be used for opening and modifying the images data
        :param name: Name of the dataset
        """
        self.list_inputs = self.__check_null_entry(list_inputs)
        self.list_labels = self.__check_null_entry(list_labels)
        self.list_additional_data = self.__check_null_entry(list_additional_data)
        self.list_data = list_inputs + list_labels + list_additional_data
        self.cv_library = cv_library
        self.transform_manager = transform_manager
        self.number_raw_instances = self.__compute_number_raw_instances()
        self.data = None
        self.use_raw_data = use_raw_data
        self.len_data = None
        self.name = name
        # Check that the given data are in a correct format before any training
        # self.__check_data()
        self.warning_video = None

        if cv_library == DEEP_LIB_OPENCV:
            try:
                Notification(DEEP_NOTIF_INFO, "OPENCV picked as Computer Vision library")
                global cv2
                import cv2
            except ImportError as e:
                Notification(DEEP_NOTIF_ERROR, str(e))
        elif cv_library == DEEP_LIB_PIL:
            try:
                Notification(DEEP_NOTIF_INFO, "PILLOW picked as Computer Vision library")
                global Image
                from PIL import Image
            except ImportError as e:
                Notification(DEEP_NOTIF_ERROR, str(e))
        else:
            Notification(DEEP_NOTIF_ERROR, "Unknown CV library flag %s" % cv_library)

    def __getitem__(self, index : int):

        """
        AUTHORS:
        --------

        author: Alix Leroy

        DESCRIPTION:
        ---------

        Get the instance (input, label, additional_data) corresponding to the given index

        PARAMETERS:
        -----------

        :param index -> Integer : Index of the instance to load


        RETURN:
        -------

        :return : Loaded and possibly transformed instance to be given to the training
        """
        inputs = []
        labels = []
        additional_data = []

        # Index of the raw data is used only to get the path to original data. Real index is used for data transformation
        index_raw_data = index % self.number_raw_instances

        # If we ask for a not existing index we use the modulo and consider the data to have to be augmented
        if index >= self.number_raw_instances:
            augment = True

        # If we ask for a raw data, augment it only if required by the user
        else:
            augment = not self.use_raw_data

        if index >= self.len_data:
            Notification(DEEP_NOTIF_FATAL, "The given instance index is too high : " + str(index))


        # Extract lists of raw data from the pandas DataFrame for the select index
        if not self.list_labels:
            if not self.list_additional_data:
                inputs = self.data.iloc[index_raw_data]
                inputs = self.__load_data(data=inputs[0], augment=augment, index=index, entry_type=DEEP_ENTRY_INPUT)         #Keep key == 0 else the datafram  also returns the name of the column (issue only on single column dataframe)
            else:
                inputs, additional_data = self.data.iloc[index_raw_data]
                inputs = self.__load_data(data=inputs, augment=augment, index=index, entry_type=DEEP_ENTRY_INPUT)
                additional_data = self.__load_data(data=additional_data, augment=augment, index=index, entry_type=DEEP_ENTRY_ADDITIONAL_DATA)
        else:
            if not self.list_additional_data:
                inputs, labels = self.data.iloc[index_raw_data]
                inputs = self.__load_data(data=inputs, augment=augment, index=index, entry_type=DEEP_ENTRY_INPUT)
                labels = self.__load_data(data=labels, augment=augment, index=index, entry_type=DEEP_ENTRY_LABEL)
            else:
                inputs, labels, additional_data = self.data.iloc[index_raw_data]
                inputs = self.__load_data(data=inputs, augment=augment, index=index, entry_type=DEEP_ENTRY_INPUT)
                labels = self.__load_data(data=labels, augment=augment, index=index, entry_type=DEEP_ENTRY_LABEL)
                additional_data = self.__load_data(data=additional_data, augment=augment, index=index,  entry_type=DEEP_ENTRY_ADDITIONAL_DATA)

        return inputs, labels, additional_data

    def __len__(self) -> int:
        """
        AUTHORS:
        --------

        author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the size of the dataset

        PARAMETERS:
        ----------

        None

        RETURN:
        -------

        return -> Integer : Length of the dataset
        """

        # If the lenth of the dataset has never been calculated we compute it
        if self.len_data == None:
            return len(self.data)

        # Else, if the length of the dataset has been specified by an external demand (specific number of instance) we return this number
        else:
            return self.len_data

    def summary(self) -> None:
        """
        AUTHORS:
        --------

        author: Alix Leroy

        DESCRIPTION:
        ------------

        Print the summary of the dataset

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        None

        """

        Notification(DEEP_NOTIF_INFO, "Summary of the '" + str(self.name)+ "' dataset : \n" + str(self.data))




    def load(self)-> None:
        """
        AUTHORS:
        --------

        author: Alix Leroy

        DESCRIPTION:
        ------------

        Load the dataset into memory

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        None

        :return:
        """


        # Read the data given as input
        inputs = self.__read_data(self.list_inputs)
        labels = self.__read_data(self.list_labels)
        additional_data = self.__read_data(self.list_additional_data)

        # Create a dictionary containing inputs, labels and additional data
        if not labels:
            if not additional_data:
                d = {'inputs': inputs}
            else:
                d = {'inputs': inputs, '_additional_data': additional_data}
        else:
            if not additional_data:
                d = {'inputs': inputs, 'labels': labels}
            else:
                d = {'inputs': inputs, 'labels': labels, 'additional_data': additional_data}


        # Convert the dictionary of data into a panda DataFrame
        try :
            self.data = pd.DataFrame(d)
        except ValueError as e:
            error_entry_array_size(d, e)




        # Update the number of instances in the DataFrame
        self.len_data = self.__len__()

        # Notice the user that the Dataset has been loaded
        Notification(DEEP_NOTIF_SUCCESS, "The '" + str(self.name) + "' dataset has successfully been loaded !")


    """
    "
    " PRIVATE METHODS
    "
    """


    def __read_data(self, list_f_data):
        """
        AUTHORS:
        --------

        author: Alix Leroy
        author:  Samuel Westlake

        DESCRIPTION:
        ------------

        Read the content given in the input files or folders

        PARAMETERS:
        -----------

        :param list_f_data : List of files or folders

        RETURN:
        -------

        :return final_data: The content of the files and folder given as input. The list is formatted to fit a pandas Dataframe columns
        """


        data = []

        # For all the files/folder given as input
        for i, f_data in enumerate(list_f_data):
            content = []

            # If the input given is a list of inputs to extend
            if type(f_data) is list:

                # For each input in the list we collect the data and extend the list
                for j, f in enumerate(f_data):
                    content.extend(self.__get_content(f))
                data.append(content)


            # If the input given is a single input
            else:
                content = self.__get_content(f_data)
                data.append(content)  # Add the new content to the list of data

        # Format the data to the format accepted by deeplodocus where final_data[i][j] = data[j][i]
        final_data = []

        if len(data) > 0:
            for i in range(len(data[0])):
                temp_data = []
                for j in range(len(data)):
                    try:
                        temp_data.append(data[j][i])
                    except IndexError as e:
                        Notification(DEEP_NOTIF_FATAL, "All your entries do not have the same number of instances : " + str(e))
                final_data.append(temp_data)
        return final_data


    def __get_content(self, f):
        """
        AUTHORS:
        --------

        author: Alix Leroy

        DESCRIPTION:
        ------------

        List all the data from a file or from a folder

        PARAMETERS:
        -----------
        :param f: A file or a folder

        RETURN:
        -------

        :return content: Content of the file/folder in a list
        """

        # Get the source path type
        source_type = self.__source_path_type(f)

        content = None

        # If f is a file
        if source_type == DEEP_TYPE_FILE:

            with open(f) as f:  # Read the file and get the data
                content = f.readlines()

            content = [x.strip() for x in content]  # Remove the end of line \n

        # If f is a folder
        elif source_type == DEEP_TYPE_FOLDER:  # If it is a folder given as input
            content = self.__get_file_paths(f)

        # Else (neither a file nor a folder)
        else:
            Notification(DEEP_NOTIF_FATAL, "The source type of the following source path does not exist : " + str(f))

        return content


    def __get_file_paths(self, directory):
        """
        AUTHORS:
        --------

        author: Samuel Westlake
        author: Alix Leroy

        DESCRIPTION:
        ------------

        Get the list of paths to every file within the given directories

        PARAMETERS:
        -----------

        :param directory: str or list of str: path to directories to get paths from

        RETURN:
        -------

        :return list of str: list of paths to every file within the given directories

        """

        paths = []

        # For each item in the directory
        for item in os.listdir(directory):
            sub_path = "%s/%s" % (directory, item)

            # If the subpath of the item is a directory we apply the self function recursively
            if os.path.isdir(sub_path):
                paths.extend(self.__get_file_paths(sub_path))

            # Else we add the path of the file to the list of files
            else:
                paths.extend([sub_path])

        return sorted_nicely(paths)


    def shuffle(self, method:int) -> None:
        """
        AUTHORS:
        --------

        author: Alix Leroy

        DESCRIPTION:
        ------------

        Shuffle the dataframe containing the data

        PARAMETERS:
        -----------
        None

        RETURN:
        -------

        :return None:
        """

        if method == DEEP_SHUFFLE_ALL:
            try:
                self.data = self.data.sample(frac=1).reset_index(drop=True)
            except:
                Notification(DEEP_NOTIF_ERROR, "Cannot shuffle the dataset")
        else:
            Notification(DEEP_NOTIF_ERROR, "The shuffling method does not exist.")

        # Reset the TransformManager
        self.reset()


    def reset(self)->None:
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


    def __load_data(self, data, augment, index, entry_type, entry_num = None):
        """
        AUTHORS:
        --------

        author: Alix Leroy

        DESCRIPTION:
        ------------

        Load (and transform is needed) the requested data

        PARAMETERS:
        -----------

        :param data: The path to the data which has to be loaded
        :param augment: Whether to augment or not the requested data
        :param index: The index of the data
        :param entry_type: Whether it in an input, a label or an additional_data
        :param entry_num: Number of the entry (input1, input2, ...) (useful for sequences)

        RETURN:
        -------

        :return loaded_data: The loaded (and transformed if required) data
        """



        loaded_data = []

        for i, d in enumerate(data):            # For each data given in the list (list = one instance of each file)
            if d is not None:
                type_data = self.__data_type(d)

                # TODO : Check how sequence behaves
                # If data is a sequence we use the function in a recursive fashion
                if type_data == DEEP_TYPE_SEQUENCE:
                    if entry_num is None:
                        entry_num = i
                    sequence_raw_data = d.split() # Generate a list from the sequence
                    loaded_data.append(self.__load_data(data=sequence_raw_data, augment=augment, index=index, entry_type=entry_type, entry_num=entry_num)) # Get the content of the list

                # Image
                elif type_data == DEEP_TYPE_IMAGE:
                    image = self.__load_image(d)

                    if self.cv_library == DEEP_LIB_PIL:
                        image = np.array(image)

                    if entry_num is None:
                        entry_num = i

                    if augment is True :
                        image = self.transform_manager.transform(data = image, index=index, type_data = type_data, entry_type = entry_type, entry_num = entry_num)

                    image = np.swapaxes(image, 0, 2)
                    loaded_data.append(image)

                # TODO : Check how video behaves
                # Video
                elif type_data == DEEP_TYPE_VIDEO:
                    video = self.__load_video(d)
                    if entry_num is None:
                        entry_num = i

                    if augment is True:
                        video = self.transform_manager.transform(data = video, index=index, type_data = type_data, entry_type = entry_type, entry_num = entry_num)
                    loaded_data.append(video)

                # Integer
                elif type_data == DEEP_TYPE_INTEGER:
                    integer = int(d)
                    loaded_data.append(integer)

                # Float
                elif type_data == DEEP_TYPE_FLOAT:
                    floating = float(d)
                    loaded_data.append(floating)

                # Numpy array
                elif type_data == DEEP_TYPE_NP_ARRAY:
                    loaded_data.append(np.load(d))

                # Data type not recognized
                else:
                    Notification(DEEP_NOTIF_FATAL,
                                 "The following data could not be loaded because its type is not recognize : %s.\n"
                                 "Please check the documentation online to see the supported types" % data)

                entry_num = None

            # If the data is None
            else:
                Notification(DEEP_NOTIF_FATAL, "The following data is None : %s" % d)


        return loaded_data



    """
    "
    " DATA TYPE ANALYZERS
    "
    """

    def __source_path_type(self, f)->int:
        """
        AUTHORS:
        --------

        author: Alix Leroy

        DESCRIPTION:
        ------------

        Find the type of the source path

        PARAMETERS:
        -----------

        A source path

        RETURN:
        -------

        :return type-> int: A type flag
        """

        # If the source path is a file
        if os.path.isfile(f):
            type = DEEP_TYPE_FILE

        # If the source path is a directory
        elif os.path.isdir(f):
            type = DEEP_TYPE_FOLDER

        # TODO: Add database as source path

        # Else
        else:
            Notification(DEEP_NOTIF_FATAL,
                         "The source type of the following source path does not exist : " + str(f))

        return type

    def __data_type(self, data)-> int:
        """
        AUTHORS:
        --------

        author: Alix Leroy

        DESCRIPTION:
        ------------

        Find the type of the given data

        PARAMETERS:
        -----------

        :param data: The data to analyze

        RETURN:
        -------

        :return: The integer flag of the corresponding type
        """

        try:
            mime = mimetypes.guess_type(data)
            mime = mime[0].split("/")[0]
        except:
            mime = None

        # Image
        if mime == "image":
            return DEEP_TYPE_IMAGE

        # Video
        elif mime == "video":
            return DEEP_TYPE_VIDEO

        # Float
        elif get_int_or_float(data) == DEEP_TYPE_FLOAT:
            return DEEP_TYPE_FLOAT

        # Integer
        elif get_int_or_float(data) == DEEP_TYPE_INTEGER:
            return DEEP_TYPE_INTEGER

        # List
        elif type(data) is list:
            return DEEP_TYPE_SEQUENCE

        if is_np_array(data) is True:
            return DEEP_TYPE_NP_ARRAY
        # Type not handled
        else:
            Notification(DEEP_NOTIF_FATAL, "The type of the following data is not handled : " + str(data))





    #
    # DATA LOADERS
    #
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

        :param image_path->str: The path of the image to load

        RETURN:
        -------

        :return: The loaded image
        """
        if self.cv_library == DEEP_LIB_OPENCV:
            image =  cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            # Check that the image was correctly loaded
            if image is None:
                Notification(DEEP_NOTIF_FATAL, "The following image cannot be loaded with OpenCV: " +str(image_path))

            # If the image is not a grayscale (only width + height axis)
            if len(image.shape) > 2:
                # Convert to RGB(a)
                image = self.__convert_bgra2rgba(image)


        elif self.cv_library == DEEP_LIB_PIL:
            try:
                image = Image.open(image_path)
            except:
                Notification(DEEP_NOTIF_FATAL, "The following image cannot be loaded with PIL: " + str(image_path))
        else:
            Notification(DEEP_NOTIF_FATAL, "The following image module is not implemented : "+ str(self.cv_library))

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

        # Convert BGR(A) to RGB(A)
        _, _, channels = image.shape

        # Handle BGR and BGRA images
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
        if self.cv_library == DEEP_LIB_OPENCV:

            # try to load the file
            try:
                cap = cv2.VideoCapture(video_path)      #Open the video

                while (cap.isOpened()):                 # While there is another frame
                    ret, frame = cap.read()
                    video.append(frame)                 # Add the frame to the sequence

            # If there is any problem during the opening of the file
            except:
                raise ValueError("An error occured while loading the following vidoe : " + str(video_path))

        # If the selected computer vision library is not compatible with we still try to open the video using OpenCV (Not optimized)
        else:
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)      #Open the video

                while (cap.isOpened()):                 # While there is another frame
                    ret, frame = cap.read()
                    video.append(frame)                 # Add the frame to the sequence

            except:
                Notification(DEEP_NOTIF_FATAL, "The following file could not be loaded : " + str(video_path) + "\n The selected Computer Vision library does not handle the videos. \n Deeplodocus tried to use OpenCV by default without success.")
        return  video # Return the sequence of frames loaded


    def __throw_warning_video(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Warn the user of the unsupported vidoe mode.

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        if self.warning_video is None:
            Notification(DEEP_NOTIF_WARNING, "The video mode is not fully supported. We deeply suggest you to use sequences of images.")
            self.warning_video = 1

    #
    # DATA CHECKERS
    #

    def __check_null_entry(self, entry):
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
        except:
            Notification(DEEP_NOTIF_ERROR, "Please check the following entry format : " + str(entry))

    def __check_data(self):
        """
        Author : Alix Leroy
        Check the validity of the data given as inputs
        :return:
        """

        Notification(DEEP_NOTIF_INFO, "Checking the data ...")


        # Check the number of data
        self.__check_data_num_instances()


        # Check the type of the data
        # TODO : Add a progress bar




        # Check data is available
        # TODO : Add a progress bar

        Notification(DEEP_NOTIF_SUCCESS, "Data checked without any error.")

    def __check_data_num_instances(self):

        # TODO : Add a progress bar
        # For each file check if we have the same number of row
        for f_data in self.list_data:

            num_instances = 0

            # If the input given is a list of inputs
            if type(f_data) is list:

                # For each input in the list we collect the data and extend the list
                for j, f in enumerate(f_data):
                    num_instances += self.__compute_number_instances(f)

            # If the input given is a single input
            else:
                num_instances = self.__compute_number_instances(f_data)


            if num_instances != self.number_instances:
                Notification(DEEP_NOTIF_FATAL, "Number of instances in " + str(self.list_inputs[0]) + " and " + str(f) + " do not match.")


    def __check_data_type(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check the type of the entries

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """

        # TODO : Add a progress bar
        # For each file check if we have the same number of row
        for f in self.list_data:

            # If the input is a file
            if self.__source_path_type(f) == DEEP_TYPE_FILE:

                with open(f) as file:
                    Notification(DEEP_NOTIF_ERROR, "Check data type not implemented")

            # If the input is a folder
            elif self.__source_path_type(f) == DEEP_TYPE_FOLDER:
                Notification(DEEP_NOTIF_FATAL, "Cannot currently check folders")

            # If it is not a file neither a folder then BUG :(
            else:
                Notification(DEEP_NOTIF_FATAL, "The following path is neither a file nor a folder : " + str(f) + ".")




    #
    # DATA UTILS
    #

    def __compute_number_raw_instances(self):
        """
        Author: Alix Leroy
        Compute the theoretical number of instances in each epoch
        The first given file/folder stands as the frame to count
        :return: theoretical number of instances in each epoch
        """
        num_instances = 0

        # If the input given is a list of inputs
        if type(self.list_inputs[0]) is list:

            # For each input in the list we collect the data and extend the list
            for j, f in enumerate(self.list_inputs[0]):
                num_instances += self.__get_number_instances(f)

        # If the input given is a single input
        else:
            num_instances = self.__get_number_instances(self.list_inputs[0])

        return num_instances





    """
    "
    " SETTERS
    "
    """

    def set_len_dataset(self, length_data : int) -> None:
        """
        AUTHORS:
        --------

        author: Alix Leroy

        DESCRIPTION:
        ------------

        Set the length of the dataset

        PARAMETERS:
        -----------

        :param length_data -> Integer:  Desired length of the dataset

        RETURN:
        -------

        None
        """

        # If the given length is smaller than the number of instances already available in the dataset
        if length_data < len(self.data):
            res = None

            # Ask the user to confirm the given length
            while res.lower() != "y" or res != "n":
                res = Notification(DEEP_NOTIF_INPUT, "Dataset contains {0} instances, are you sure you want to only use {1} instances ? (Y/N) ".format(len(self.data), length_data))

            if res.lower() == "y":
                self.len_data = length_data

            # If not confirmed, keep the current size of the dataset as default
            else:
                self.len_data = len(self.data)


        # If there isn't any issue of length set the length given as argument
        else:
            self.len_data = length_data




    def set_use_raw_data(self, use_raw_data: bool) -> None:
        """
        AUTHORS:
        --------

        author: Alix Leroy

        DESCRIPTION:
        ------------

        Set the use_raw_data attribut to new value

        PARAMETERS:
        -----------

        :param use_raw_data -> bool : Whether to use or not the raw data in the training


        RETURN:
        -------

        None
        """

        self.use_raw_data = use_raw_data


    """
    "
    " GETTERS
    "
    """


    def __get_number_instances(self, f):
        """
        Authors : Alix Leroy,
        Get the number of instances in a file or a folder
        :param f: A file or folder path
        :return: Number of instances in the file or the folder
        """

        # If the frame input is a file
        if self.__source_path_type(f) == DEEP_TYPE_FILE:

            with open(f) as f:
                num_instances = sum(1 for _ in f)

        # If the frame input is a folder
        elif self.__source_path_type(f) == DEEP_TYPE_FOLDER:

            raise ValueError("Not implemented")

        # If it is not a file neither a folder then BUG :(
        else:
            Notification(DEEP_NOTIF_FATAL, "The following input is neither a file nor a folder :" + str(f))

        return num_instances