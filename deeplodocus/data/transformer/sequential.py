from deeplodocus.data.transformer.transformer import Transformer
from deeplodocus.utils.notification import Notification, DEEP_FATAL


class Sequential(Transformer):

    def __init__(self, augmentation_functions=None, random_order=False):
        Transformer.__init__(self)

        self.augmentation_functions = augmentation_functions
        self.random_order = random_order


    def augment_image(self, image):

        for key in self.augmentation_functions:

            image = self.__augment_image(image, key)

        return image


    def transform(self, data, index, data_type):
        """
        Authors : Alix Leroy,
        :param data: data to transform
        :param index: The index of the instance in the Data Frame
        :param data_type: The type of data
        :return: The transformed data
        """

        for augmentation in self.augmentation_functions:
            if data_type == "image":
                data_transformed = self.__transform_image(data, index, augmentation)
            elif data_type =="video":
                data_transformed = self.__transform_video(data, index, augmentation)
            else:
                Notification(DEEP_FATAL, "The following data type is not supported for data transformation : "  + str(data_type))


        return data_transformed