import pandas as pd
import numpy as np
import os
import mimetypes

from deeplodocus.utils.generic_utils import get_file_paths
from deeplodocus.utils.notification import Notification,  DEEP_FATAL, DEEP_SUCCESS, DEEP_ERROR, DEEP_INFO

cv_library = "PIL"

# IMPORT COMPUTER VISION LIBRARY
if cv_library == "opencv":
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV could not be loaded. Please check https://opencv.org/ ")

elif cv_library == "PIL":
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL Image could not be loaded. "
                          "Please check if PIL is correctly installed. "
                          "If not use 'pip install Pillow'.")
else:
    Notification(DEEP_FATAL,  "The following image module is not implemented : %s" % cv_library)


class DataSet(object):

    def __init__(self, list_inputs, list_labels, list_additional_data, transform=None, cv_library="PIL", write_logs=True):
        """
        :param list_inputs: list of files/folder/list of files/folders
        :param list_labels: list of files/folder/list of files/folders
        :param list_additional_data: list of files/folder/list of files/folders
        :param root_dir:
        :param transform:
        :param cv_library: The computer vision library we want to use
        """
        self.list_inputs = list_inputs
        self.list_labels = list_labels
        self.list_additional_data = list_additional_data
        self.list_data = list_inputs + list_labels + list_additional_data
        self.cv_library = cv_library
        self.transform = transform
        self.write_logs = write_logs
        self.number_instances = self.__compute_number_instances()
        self.data = None
        self.include_raw_data = True
        # Check that the given data are in a correct format before any training
        #self.__check_data()

    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        # Extract lists of raw data from the pandas DataFrame for the select index
        if not self.list_labels:
            if not self.list_additional_data:
                inputs = self.data.iloc[index]
                inputs = self.__load_data(inputs, index, "inputs")
                return inputs
            else:
                inputs, additional_data = self.data.iloc[index]
                inputs = self.__load_data(inputs, index, "inputs")
                additional_data = self.__load_data(additional_data, index, "additional_data")
                return inputs, additional_data
        else:
            if not self.list_additional_data:
                inputs, labels = self.data.iloc[index]
                inputs = self.__load_data(inputs, index, "inputs")
                labels = self.__load_data(labels, index, "labels")
                return inputs, labels
            else:
                inputs, labels, additional_data = self.data.iloc[index]
                inputs = self.__load_data(inputs, index, "inputs")
                labels = self.__load_data(labels, index, "labels")
                additional_data = self.__load_data(additional_data, index, "additional_data")
                return inputs, labels, additional_data

    def __len__(self):
        """
        Authors : Alix Leroy,
        Get the number of instances in one epoch (augmentation included)
        :return:
        """
        return len(self.data)

    def summary(self):
        """
        Authors : Alix Leroy,
        Print the summary of the dataset
        :return: None
        """

        Notification(DEEP_INFO, "SUMMARY OF THE DATASET : \n" + str(self.data), write_logs=self.write_logs)

    def set_include_raw_data(self, include):
        """
        Authors : Alix Leroy
        Whether raw data should be included or only use transform
        :param include: Boolean
        :return:
        """
        self.include_raw_data = include

    def has_labels(self):
        """
        Authors : Alix Leroy,
        Check whether the dataset contains labels
        :return: Boolean
        """

        columns = list(self.data.columns.values)

        if "labels" in columns:
            return True
        else:
            return False

    def has_additional_data(self):
        """
        Authors : Alix Leroy,
        Check whether the dataset contains additional_data
        :return: Boolean
        """

        columns = list(self.data.columns.values)

        if "additional_data" in columns:
            return True
        else:
            return False

    def fill_data(self):
        """
        Authors
        :return:
        """
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
        self.data = pd.DataFrame(d)


    def __read_data(self, list_f_data):
        """
        Authors : Alix Leroy, Samuel Westlake
        :param list_f_data: All the data given in the config file
        :return: Data formatted to fill a pandas dataframe
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
                    temp_data.append(data[j][i])
                final_data.append(temp_data)
        return final_data

    def __get_content(self, f):
        """
        Authors : Alix Leroy
        List all the data from a file or a folder
        :param file: a file path
        :return: Content of the file in a list
        """

        # If it is a file
        if self.__type_input(f) == "file":

            with open(f) as f:  # Read the file and get the data
                content = f.readlines()

            content = [x.strip() for x in content]  # Remove the end of line \n

        elif self.__type_input(f) == "folder":  # If it is a folder given as input
            content = get_file_paths(f)

        else:
            raise ValueError("Not implemented")

        return content

    def __shuffle(self):
        """
        Authors : Alix Leroy,
        Shuffle the dataframe containing the dataset
        :return: None
        """
        try :
            self.data = self.data.sample(frac=1).reset_index(drop=True)
        except:
            Notification(DEEP_ERROR, "Could not shuffle the dataset", write_logs=self.write_logs)


    def __load_data(self, data, index, entry_type, entry_num = None):
        """
        :param data: The data to load in memory
        :param entry_type : Whether it in an input, a label or an additional_data
        :return: The data loaded and transformed if needed
        """
        loaded_data = []
        for i, d in enumerate(data):            # For each data given in the list (list = one instance of each file)
            if d is not None:
                type_data = self.__data_type(d)

                # If data is a sequence we use the function in a recursive fashion
                if type_data == "sequence":
                    if entry_num is None:
                        entry_num = i
                    sequence_raw_data = d.split() # Generate a list from the sequence
                    loaded_data.append(self.__load_data(sequence_raw_data, index, entry_type, entry_num)) # Get the content of the list

                # Image
                elif type_data == "image":
                    image = self.__load_image(d)
                    if entry_num is None:
                        entry_num = i
                    if self.transform is not None:
                        image = self.transform.transform(data = image, index=index, type_data = type_data, entry_type = entry_type, entry_num = entry_num)
                    loaded_data.append(image)

                # Video
                elif type_data == "video":
                    video = self.__load_video(d)
                    if entry_num is None:
                        entry_num = i
                    if self.transform is not None:
                        video = self.transform.transform(data = video, index=index, type_data = type_data, entry_type = entry_type, entry_num = entry_num)
                    loaded_data.append(video)

                # Integer
                elif type_data == "integer":
                    integer = int(d)
                    loaded_data.append(integer)

                # Float
                elif type_data == "float":
                    floating = float(d)
                    loaded_data.append(floating)

                # Data type not recognized
                else:
                    raise ValueError("The following data could not be loaded because its type is not recognize : %s.\n"
                                     "Please check the documentation online to see the supported types" % data)

                entry_num = None
            else:
                raise ValueError("The following data is None : %s" % d)


        return loaded_data

    #
    # DATA TYPE ANALYZERS
    #

    def __type_input(self, f):
        """
        :param f:
        :return:
        """
        if os.path.isfile(f):
            type = "file"
        elif os.path.isdir(f):
            type = "folder"
        else:
            raise ValueError("The following data file/folder could not be recognize : " +str (f))
        return type

    def __data_type(self, data):
        """
        Authors : Alix Leroy,
        Get the type of data given
        :param data:
        :return:
        """

        try:
            mime = mimetypes.guess_type(data)
            mime = mime[0].split("/")[0]
        except:
            mime = ""

        # Image
        if mime == "image":
            return "image"

        # Video
        elif mime == "video":
            return "video"

        # Float
        elif self.__get_int_or_float(data) == "float":
            return "float"

        # Integer
        elif self.__get_int_or_float(data) == "int":
            return "integer"

        # List
        elif type(data) is list:
            return "sequence"

        # Type not handled
        else:
            Notification(DEEP_FATAL, "The following type of data is not handled : " + str(data), write_logs=self.write_logs)



    @staticmethod
    def __get_int_or_float(v):
        try:
            number_as_float = float(v)
            number_as_int = int(number_as_float)
            return "int" if number_as_float == number_as_int else "float"
        except ValueError:
            return False

    #
    # DATA LOADERS
    #

    def __load_image(self, image_path):
        """
        Authors : Alix Leroy,
        :param image_path: The path of the image to load
        :return: The loaded image
        """
        if self.cv_library == "opencv":
            image =  cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)

            # Check that the image was correctly loaded
            if not image is None:
                Notification(DEEP_FATAL, "The following image cannot be loaded : " +str(image_path), write_logs=self.write_logs)

            # Convert to RGB(a)
            image = self.__convert_bgra2rgba(image)

        elif self.cv_library == "PIL":
            try:
                image = Image.open(image_path)
            except:
                Notification(DEEP_FATAL, "The following image cannot be loaded : " +str(image_path), write_logs=self.write_logs)

        else:
            Notification(DEEP_FATAL, "The following image module is not implemented : "+ str(self.cv_library), write_logs=self.write_logs)

        return image

    def __convert_bgra2rgba(self, image):
        """
        Authors : Alix Leroy,
        Convert BGR(alpha) image to RGB(alpha) image
        :param image: image to convert
        :return: a RGB(alpha) image
        """

        # Convert BGR(A) to RGB(A)
        _, _, channels = image.shape
        if channels == 3:
            image = image[:, :, (2, 1, 0)]
        elif channels == 4:
            image = image[:, :, (2, 1, 0, 3)]

        return image


    def __load_video(self, video_path):
        """
        Author : Alix Leroy
        :param video_path: absolute path to a video
        :return: a list of frame from the video
        """

        video = []

        # If the computer vision library selected is OpenCV
        if self.cv_library == "opencv":

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
                Notification(DEEP_FATAL, "The following file could not be loaded : " + str(video_path) + "\n The selected Computer Vision library does not handle the videos. \n Deeplodocus tried to use OpenCV by default without success.", write_logs=self.write_logs)
        return  video # Return the sequence of frames loaded




    #
    # DATA CHECKERS
    #

    def __check_data(self):
        """
        Author : Alix Leroy
        Check the validity of the data given as inputs
        :return:
        """

        Notification(DEEP_INFO, "Checking the data ...", write_logs=self.write_logs)


        # Check the number of data
        self.__check_data_num_instances()


        # Check the type of the data
        # TODO : Add a progress bar




        # Check data is available
        # TODO : Add a progress bar

        Notification(DEEP_SUCCESS, "Data checked without any error.", write_logs=self.write_logs)

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
                Notification(DEEP_FATAL, "Number of instances in " + str(self.list_inputs[0]) + " and " + str(f) + " do not match.", write_logs=self.write_logs)


    def __check_data_type(self):

        # TODO : Add a progress bar
        # For each file check if we have the same number of row
        for f in self.list_data:

            # If the input is a file
            if self.__type_input(f) == "file":

                with open(f) as file:
                    Notification(DEEP_ERROR, "Check data type not implemented", write_logs=self.write_logs)


            # If the input is a folder
            elif self.__type_input(f) == "folder":
                Notification(DEEP_FATAL, "Cannot currently check folders", write_logs=self.write_logs)


            # If it is not a file neither a folder then BUG :(
            else:
                Notification(DEEP_FATAL, "The following path is neither a file nor a folder : " + str(f) + ".", write_logs=self.write_logs)




    #
    # DATA UTILS
    #

    def __compute_number_instances(self):
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


    def __get_number_instances(self, f):
        """
        Authors : Alix Leroy,
        Get the number of instances in a file or a folder
        :param f: A file or folder path
        :return: Number of instances in the file or the folder
        """

        # If the frame input is a file
        if self.__type_input(f) == "file":

            with open(f) as f:
                num_instances = sum(1 for _ in f)

        # If the frame input is a folder
        elif self.__type_input(f) == "folder":

            raise ValueError("Not implemented")

        # If it is not a file neither a folder then BUG :(
        else:
            Notification(DEEP_FATAL, "The following input is neither a file nor a folder :" + str(f), write_logs=self.write_logs)

        return num_instances



