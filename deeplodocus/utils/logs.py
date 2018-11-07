import os
import datetime
import __main__

class Logs(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    A class which manages the logs
    """

    def __init__(self, type:str)->None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize a log object

        PARAMETERS:
        -----------

        :param type->str: The log type

        RETURN:
        -------

        :return: None
        """
        self.type = type

    def check_init(self)->None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check that the log can be initialized without issue with a previous version

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """

        # Log paths
        logs_folder_path = os.path.dirname(__main__.__file__) + "/logs"
        logs_file_path = logs_folder_path + "/" + str(self.type) + ".logs"

        # Check the logs folder exists
        self.__check_log_folder_exists(logs_folder_path)

        # Check the log file exists
        self.__check_log_file_exists(logs_file_path)

    def add(self, text:str)->None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Add a line to the log

        PARAMETERS:
        -----------

        :param text->str: The text to add

        RETURN:
        -------

        :return: None

        """

        logs_path = os.path.dirname(__main__.__file__) + "/logs/" + str(self.type) + ".logs"
        timestr = datetime.datetime.now()

        with open(logs_path, "a") as log:
            log.write(str(timestr) + " : " + text + "\n")



    def __check_log_folder_exists(self, logs_path:str):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check if the log folder exists and create it if not.

        PARAMETERS:
        -----------

        :param logs_path->str: The absolute path to the folder

        RETURN:
        -------

        :return: None
        """

        # If the folder path does not exist we create it
        if not os.path.exists(logs_path):
            try:
                os.mkdir(logs_path)

            # Raise Value (Cannot use Notification from Logs class)
            except:
                raise ValueError("An error occurred during the generation of the LOGS folder. "
                                 "Make sure the framework has the permission to write in this folder")


    def __check_log_file_exists(self, logs_path:str):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check if the corresponding log file exists

        PARAMETERS:
        -----------
        :param logs_path->str: The absolute path to the log file

        RETURN:
        -------

        :return: None
        """

        # If the file path does exist we finish the previous log
        if  os.path.exists(logs_path):
            self.close_log()

        # Create the log file
        try:
            self.__create_log_file(logs_path)

        # Raise Value (Cannot use Notification from Logs class)
        except:
            raise ValueError("An error occurred during the generation of the log folder. "
                             "Make sure the framework as the permission to write in this folder")

    def __create_log_file(self, logs_path:str)->None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Create the log file and insert the date time on first line

        PARAMETERS:
        -----------
        :param logs_path->str: The path to the log file

        RETURN:
        -------

        :return: None
        """
        # Initialize the log file
        open(logs_path, "w").close()

        # Append the date time as first line
        with open(logs_path, "a") as log:
            timestr = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log.write("Initialized : " + str(timestr) + "\n")

    def close_log(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Terminates the current log

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """

        logs_path = os.path.dirname(__main__.__file__)+ "/logs/" + str(self.type) +".logs"


        # If the log file exists
        if os.path.isfile(logs_path):
            # Read the first line
            with open(logs_path, "r") as f:
                line = f.readline()

            # Get the initialization time
            time = line.split(":")[1].replace(" ", "").replace("\n", "")

            # Generate new name
            new_log_name = os.path.dirname(__main__.__file__)+ "/logs/" + str(self.type) + "_" + str(time) + ".logs"

            os.rename(logs_path, new_log_name)
            print("test")
