import os
import datetime
import __main__





class Logs(object):

    """
    Types : notification,
    """


    def __init__(self, type):
        """
        Authors : Alix Leroy,
        Initialize a log object
        :param type:
        """
        self.type = type

    def check_init(self):
        """
        Authors :  Alix Leroy,
        Check that the log can be initialized without issue with a previous version
        :return: None
        """

        logs_folder_path = os.path.dirname(__main__.__file__)+ "/logs"
        logs_file_path = logs_folder_path +"/" + str(self.type) +".logs"

        # Check the logs folder exists
        self.__check_log_folder_exists(logs_folder_path)

        # Check the log file exists
        self.__check_log_file_exists(logs_file_path)



    def add(self, text):
        """
        Authors : Alix Leroy,
        Add a line to the log
        :param text: The text to add
        :return: None
        """


        logs_path = os.path.dirname(__main__.__file__)+ "/logs/" + str(self.type) +".logs"
        timestr = datetime.datetime.now()

        with open(logs_path, "a") as log:
            log.write(str(timestr) + " : " + text + "\n")



    def __check_log_folder_exists(self, logs_path):
        """
        Authors : Alix Leroy,
        Check if the log folder exists
        :return: None
        """

        if not os.path.exists(logs_path):
            try:
                os.mkdir(logs_path)
            except:
                raise ValueError("An error occurred during the generation of the LOGS folder. Make sure the framework has the permission to write in this folder")


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





    def close_log(self):
        """
        Authors : Alix Leroy,
        Terminates the current log
        :return: None
        """

        logs_path = os.path.dirname(__main__.__file__)+ "/logs/" + str(self.type) +".logs"


        # If the log file exists
        if os.path.isfile(logs_path):
            # Read the first line
            with open(logs_path, "r") as f:
                line = f.readline()

            # Get the initialization time
            time = line.split(":")[1].replace(" ", "")

            # Generate new name
            new_log_name =os.path.dirname(__main__.__file__)+ "/logs/" + str(self.type) + "_" + str(time) + ".logs"

            os.rename(logs_path, new_log_name)
