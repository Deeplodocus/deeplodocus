import cv2
import numpy as np


from deeplodocus.utils.notification import Notification, DEEP_FATAL


class Transformer(object):

    def __init__(self, path):

        self.path = path
        self.pointer_to_transformer = self.__get_pointer()

        self.last_index = None
        self.last_transforms = []



    def transform(self, data, index, data_type):
        """
        Authors : Alix Leroy,
        :param data: data to transform
        :param index: The index of the instance in the Data Frame
        :param data_type: The type of data
        :return: The transformed data
        """
        pass # Will be overridden

    def __apply_transform(self, data, transformation, parameters):
        """
        Authors : Alix Leroy
        :param data: Data to transform
        :param transformation: The transformation to do
        :param parameters: The parameters of the transformation
        :return: The transformed data
        """

        pass



    def __apply_last_transforms(self, data):
        """
        Authors : Alix Leroy
        Apply the last transform to the current instance
        :param data : The data to transform
        :return: The transformed data
        """

        # For each transformation in the list
        for transformation, parameters in self.list_last_transforms:
            data = self.__apply_transform(data, transformation, parameters)


        return data

    def __get_pointer(self):
        """
        Authors : Alix Leroy,
        Check if the transformer has to point to another transformer
        :return: The pointer to another transformer
        """
        if str(self.path)[0] == "*":                 # If we have defined the transformer as a pointer to another transformer
            path_splitted = str(self.path).split(":")[1:]

            if len(path_splitted) != 2 :
                Notification(DEEP_FATAL, "The following transformer does not point correctly to another transformer, please check the documentation : " + str(self.path))

            if self.__pointer_type_exists(path_splitted[0]) is False:
                Notification(DEEP_FATAL, "The type of the following transformer's pointer does not exist, please check the documentation : " + str(self.path))

            if isinstance(path_splitted[1], int) is False:
                Notification(DEEP_FATAL, "The second argument of the following transformer's pointer is not an integer : " + str(self.path))

            return [str(path_splitted[0]), int(path_splitted[1])] # Return type and index of the pointer

        else:
            return None

    def __pointer_type_exists(self, type):

        type = str(type).lower()

        if type == "inputs":
            return True

        elif  type == "labels":
            return True

        elif type == "additional_data":
            return True

        else :
            return False



    def __transformation_video(self, video, index):
        """
        Authors : Alix Leroy,
        :param video: The video to transform
        :param index: The index of the video in the Data Frame
        :return: The transformed video
        """

        transformed_video = []

        # For each image in the video we transform the image

        # If we want to use the transformations of a previous index
        if self.last_index == index:
            for image in video:
                transformed_image = self.__apply_last_transforms(image)
                transformed_video.append(transformed_image)
        else:
            for image in video:
                transformed_image = self.__transform(image, index)
                transformed_video.append(transformed_image)

        return transformed_video

    def __transform_image(self, image, key):

        """
        Author : Alix Leroy
        :param image: input image to augment
        :param key: the parameters of the augmentation in a dictionnary
        :return: augmented image
        """

        ################
        # ILLUMINATION #
        ################
        if key == "adjust_gamma":
            gamma = np.random.random(key["gamma"][0], key["gamma"][1])
            image = self.adjust_gamma(image, gamma)


        #########
        # BLURS #
        #########
        elif key == "average":
            kernel = tuple(int(key["kernel_size"]), int(key["kernel_size"]))
            image = cv2.blur(image, kernel)

        elif key == "gaussian_blur":
            kernel = tuple(int(key["kernel_size"]), int(key["kernel_size"]))
            image = cv2.GaussianBlur(image, kernel, 0)

        elif key == "median_blur":

            image = cv2.medianBlur(image, int(key["kernel_size"]))

        elif key == "bilateral_blur":
            diameter = int(key["diameter"])
            sigma_color = int(key["sigma_color"])
            sigma_space = int(key["sigma_space"])
            image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


        #########
        # FLIPS #
        #########
        elif key == "horizontal_flip":
            image = cv2.flip(image, 0)

        elif key == "vertical_flip":
            image = cv2.flip(image, 1)


        #############
        # ROTATIONS #
        #############

        elif key == "random_rotation":
            angle = np.random.random(00, 359.9)
            shape = image.shape
            rows, cols = shape[0:2]
            m = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            image = cv2.warpAffine(image, m, (cols, rows)).astype(np.float32)


        elif key == "boundary_rotation":
            angle = float(key["angle"])
            angle = (2 * np.random.rand() - 1) * angle
            shape = image.shape
            rows, cols = shape[0:2]
            m = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            image = cv2.warpAffine(image, m, (cols, rows)).astype(np.float32)


        elif key == "rotation":
            angle = float(key["angle"])
            shape = image.shape
            rows, cols = shape[0:2]
            m = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            image = cv2.warpAffine(image, m, (cols, rows)).astype(np.float32)



        else:
            Notification(DEEP_FATAL, "This transformation function does not exist : " + str(transformation))
        return image



    def adjust_gamma(self, image, gamma=1.0):

        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        return cv2.LUT(image, table)


    #
    # IMAGES
    #

    def resize(image, shape, keep_aspect=True, padding=0):
        """
        Author: Samuel Westlake, Alix Leroy
        :param image: np.array, input image
        :param shape: tuple, target shape
        :param keep_aspect: bool, whether or not the aspect ration should be kept
        :param padding: int, value for padding if keep_aspect is True
        :return: np.array, image of size shape
        """

        # If we want to reduce the image
        if image.shape[0] * image.shape[1] > shape[0] * shape[1]:
            interpolation = cv2.INTER_LINEAR_EXACT  # Use the Bilinear Interpolation
        else:
            interpolation = cv2.INTER_CUBIC  # Use the Bicubic interpolation

        if keep_aspect:
            scale = min(np.asarray(shape[0:2]) / np.asarray(image.shape[0:2]))
            new_size = np.array(image.shape[0:2]) * scale
            image = cv2.resize(image, (int(new_size[1]), int(new_size[0])), interpolation=interpolation)
            image = pad(image, shape, padding)
        else:
            image = cv2.resize(image, (shape[0], shape[1]), interpolation=interpolation)
        return image.astype(np.float32)

    def pad(image, shape, value=0):
        """
        Author: Samuel Westlake and Alix Leroy
        Pads an image to self.x_size with a given value with the image centred
        :param: image: input image
        :param: value
        :return: Padded image
        """
        padded = np.empty(shape, dtype=np.uint8)
        padded.fill(value)
        y0 = int((shape[0] - image.shape[0]) / 2)
        x0 = int((shape[1] - image.shape[1]) / 2)
        y1 = y0 + image.shape[0]
        x1 = x0 + image.shape[1]

        nb_channels = padded.shape[2]

        if nb_channels == 1:
            padded[y0:y1, x0:x1, 0] = image
        else:

            padded[y0:y1, x0:x1, :] = image

        return padded.astype(np.float32)

    def random_channel_shift(image, shift):
        shift = np.random.randint(-shift, shift, image.shape[2])
        for ch in range(image.shape[2]):
            image[:, :, ch] += shift[ch]
        image[image < 0] = 0
        image[image > 255] = 255
        return image.astype(np.float32)

    def random_rotate(image, angle):
        angle = (2 * np.random.rand() - 1) * angle
        shape = image.shape
        rows, cols = shape[0:2]
        m = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        return cv2.warpAffine(image, m, (cols, rows)).astype(np.float32)

    def rotate(image, angle):
        shape = image.shape
        rows, cols = shape[0:2]
        m = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        return cv2.warpAffine(image, m, (cols, rows)).astype(np.float32)



    #
    # DATA NORMALIZERS
    #

    def __normalize_image(self, image):
        """
        Author : Alix Leroy
        Normalize an image (mean and standard deviation)
        :param image: an image
        :return: a normalized image
        """


        # The normalization compute the mean of the image online.
        # This takes more time than just giving the mean as a parameter in the config file
        # However this time is still relatively small
        # Moreover this is done in parallel of the training
        # Note 1 : OpenCV is roughly 50% faster than numpy
        # Note 2 : Could be a limiting factor for big "mini"-batches (>= 1024) and big images (>= 512, 512, 3)

        # If OpenCV is selected (50% than numpy)
        if cv_library == "opencv":
            channels = image.shape[-1]
            mean = cv2.mean(image)

            normalized_image = (image - mean[:channels]) / 255  # Norm = (data - mean) / standard deviation

        # Default option
        else:
            mean = np.mean(image, axis=(0, 1))  # Compute the mean on each channel

            normalized_image = (image - mean) / 255  # Norm = (data - mean) / standard deviation

        return normalized_image


    def __normalize_video(self, video):
        """
        Author: Alix Leroy
        :param video: sequence of frames
        :return: a normalized sequence of frames
        """

        video = [self.__normalize_image(frame) for frame in video]

        return video

