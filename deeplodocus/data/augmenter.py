import os
import __main__
import random
import string
import datetime


from deeplodocus.utils.notification import Notification, DEEP_INFO, DEEP_SUCCESS, DEEP_FATAL

class Augmenter(object):


    def __init__(self, dataset, nb_augmentations, save_path = None):

        self.dataset = dataset
        self.nb_augmentation = nb_augmentations
        self.save_path = save_path
        self.dataset.set_include_raw_data(False)

        self.files_initialised = False

        self.list_inputs = []
        self.list_labels = []
        self.list_additional_data = []



    def augment(self):

        len_dataset = self.dataset.__len__()
        nb_augmentation_done = 0

        Notification(DEEP_INFO, "Starting the offline data augmentation ... ")

        while(nb_augmentation_done < self.nb_augmentation):

            for i in range(len_dataset):
                if nb_augmentation_done >= self.nb_augmentation:
                    break

                transformed_data = self.dataset.__getitem__(i)

                self.__save_data(transformed_data)

                nb_augmentation_done+=1

        Notification(DEEP_SUCCESS, "Offline data augmentation finished.")


    def __save_data(self, data):
        """
        Authors : Alix Leroy,
        Save the data
        :param data:
        :return:
        """

        if self.files_initialised is False :
            self.__initialise_files(data)




    def __initialise_files(self, data):
        """
        Authors : Alix Leroy,
        Initialise the files for data augmentation
        :param data:
        :return:
        """

        has_labels = self.dataset.has_labels()
        has_additional_data = self.dataset.has_additional_data()

        if self.save_path is None :
            self.save_path  = os.path.dirname(__main__.__file__)+ "/augmentations"

        self.__check_augmentation_folder_exists(save_path=self.save_path)

    def __check_augmentation_folder_exists(self, save_path):
        """
        Authors : Alix Leroy,
        Check if the augmentation folder exists
        :return: None
        """

        if not os.path.exists(save_path):
            try:
                os.mkdir(save_path)
            except:
                Notification(DEEP_FATAL, "An error occurred during the generation of the AUGMENTATION folder. Make sure the framework has the permission to write in this folder")


    def __generate_random_name(self):
        """
        Authors : Alix Leroy,
        Generate a random unique string as a filename
        :return: a random string
        """

        random_name  = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
        random_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_" +  random_name
        return random_name


    def __check_log_file_exists(self, logs_path):
        """
        Authors : Alix Leroy,
        Check if the corresponding log file exists
        :return: None
        """

        if  os.path.exists(logs_path):
            self.close_log()

        try:
            self.__create_log_file(logs_path)
        except:
            raise ValueError(
                "An error occurred during the generation of the log folder. Make sure the framework as the permission to write in this folder")

    def __create_log_file(self, logs_path):

        # Initialize the log file
        open(logs_path, "w").close()
        with open(logs_path, "a") as log:
            timestr = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log.write("Initialized : " + str(timestr) + "\n")


    def __end_augmentation(self):


        self.dataset.set_transformer(None)



