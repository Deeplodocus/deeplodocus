import os.path
import time


from deeplodocus.utils.notification import Notification, DEEP_ERROR, DEEP_INPUT, DEEP_SUCCESS, DEEP_FATAL, DEEP_WARNING
from deeplodocus.utils.logo import Logo
from deeplodocus.utils.end import End
from deeplodocus.utils.logs import Logs
from deeplodocus.ui.user_interface import UserInterface

class Brain(object):



    def __init__(self, config_path):

        # List of logs
        self.logs = ["notification"]
        # Initialize the logs
        self.__init_logs()

        # Version of the software
        self.version = "0.1.0"
        # Display the Deeplodocus logo
        Logo(version=self.version)

        # Defin exit flags
        self.exit_flags = ["q", "quit", "exit"]

        # Load the config
        self.config = None
        self.__get_config_path(config_path)

        self.user_interface = None




    def wake(self):
        """
        Authors : Alix Leroy
        Main of deeplodocus framework
        :return: None
        """

        while True :
            command = Notification(DEEP_INPUT, "Waiting for instruction...").get()

            command = command.replace(" ", "")
            if command in self.exit_flags:
                break

            else:
                self.__run_command(command)
                time.sleep(0.2)                                 # Sleep to make sure that asynchronous commands are completed

        if self.user_interface is not None:
            self.user_interface.stop()
        End(error = False)




    def __run_command(self, command):

        # Load a new config file
        if command == "load_config":
            self.__load_config()

        # train the network
        elif command == "train":
            print("Train")

        # Start the User Interface
        elif command == "ui" or command == "user_interface" or command == "interface":
            if self.user_interface is None:
                self.user_interface = UserInterface()
            else:
                Notification(DEEP_ERROR, "The user interface is already running")

        elif command == "ui_stop" or command == "stop_ui" or command == "ui stop":
            if self.user_interface is not None:
                self.user_interface.stop()
                self.user_interface = None

        else:
            Notification(DEEP_WARNING, "The given command does not exist.")


    def __get_config_path(self, config_path=None):

        config_path_valid = False

        while config_path_valid is False:

            if config_path is None :
                config_path = Notification("Input", "Please insert the config file path :").get()

            else:

                if config_path in self.exit_flags:
                    Notification(DEEP_FATAL, "No config file given as input")


                else:
                    if os.path.isfile(config_path):
                        self.__load_config(config_path)
                        config_path_valid = True

                    else:
                        Notification(DEEP_ERROR, "The given path does not point to a file")
                        config_path = None



    def __load_config(self, config_file):

        Notification(DEEP_SUCCESS, "Config file loaded (" +str(config_file)+")")


    def __init_logs(self):
        """
        Authors : Alix Leroy
        Initialize all logs
        :return:None
        """

        for log_name in self.logs:
            Logs(log_name).check_init()

        Notification(DEEP_SUCCESS, "Logs initialized ! ")





