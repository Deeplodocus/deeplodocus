import os.path
import time

from deeplodocus.utils.notification import Notification
from deeplodocus.utils.flags import *
from deeplodocus.utils.namespace import Namespace
from deeplodocus.utils.logo import Logo
from deeplodocus.utils.end import End
from deeplodocus.utils.logs import Logs
from deeplodocus.ui.user_interface import UserInterface
from deeplodocus import __version__


class Brain(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy
    :author: Samuel Westlake

    DESCRIPTION:
    ------------

    A Brain class that manages the commands of the user and allows to start the training
    """

    def __init__(self, config_path, write_logs:bool=True):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize a Deeplodocus Brain

        PARAMETERS:
        -----------

        :param config_path->str: The config path
        :param write_logs->bool: Whether to write logs or not

        RETURN:
        -------

        :return: None
        """

        self.write_logs = write_logs
        self.config_path = config_path
        self.logs = ["notification"]
        self.__init_logs()
        Logo(version=__version__, write_logs=write_logs)
        self.exit_flags = ["q", "quit", "exit"]
        self.config = None
        self.user_interface = None
        self.load_config()

    def wake(self):
        """
        Authors : Alix Leroy
        Main of deeplodocus framework
        :return: None
        """
        while True:
            command = Notification(DEEP_NOTIF_INPUT, "Waiting for instruction...", write_logs=self.write_logs).get()
            command = command.replace(" ", "")
            if command in self.exit_flags:
                break
            else:
                self.__run_command(command)
                time.sleep(0.5)                 # Sleep to make sure that asynchronous commands are completed
        if self.user_interface is not None:
            self.user_interface.stop()
        End(error=False)

    def load_config(self):
        """
        Author: SW
        Function: Checks current config path is valid, if not, user is prompted to give another
        :return: bool: True if a valid config path is set, otherwise, False
        """
        while True:
            if os.path.isfile(self.config_path):
                self.config = Namespace(self.config_path)
                directory = "/".join(self.config_path.split("/")[:-1])
                for key, path in self.config.get().items():
                    self.config.get()[key] = Namespace("%s/%s" % (directory, path))
                self.config.load(self.config_path, "main")
                Notification(DEEP_NOTIF_SUCCESS, "Config file loaded (%s)" % self.config_path)
                return True
            else:
                if self.config_path in self.exit_flags:
                    return False
                else:
                    Notification(DEEP_NOTIF_ERROR, "Given path does not point to a file (%s)" % self.config_path)
                    self.config_path = Notification(DEEP_NOTIF_INPUT, "Please insert the config file path :").get()

    def __run_command(self, command:str)->None:
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Run the given command

        PARAMETERS:
        -----------

        :param command->str: The given command

        RETURN:
        -------

        :return: None
        """

        # Load a new config file
        if command == "load_config":
            self.load_config()

        # train the network
        elif command == "train":
            print("Train")

        # Start the User Interface
        elif command == "ui" or command == "user_interface" or command == "interface":
            if self.user_interface is None:
                self.user_interface = UserInterface()
            else:
                Notification(DEEP_NOTIF_ERROR, "The user interface is already running", write_logs=self.write_logs)

        elif command == "ui_stop" or command == "stop_ui" or command == "ui stop":
            if self.user_interface is not None:
                self.user_interface.stop()
                self.user_interface = None

        else:
            Notification(DEEP_NOTIF_WARNING, "The given command does not exist.", write_logs=self.write_logs)

    def __init_logs(self):
        """
        Authors : Alix Leroy
        Initialize all logs
        :return:None
        """
        for log_name in self.logs:
            Logs(log_name).check_init()
        Notification(DEEP_NOTIF_SUCCESS, "Logs initialized ! ", write_logs=self.write_logs)


if __name__ == "__main__":
    import argparse

    def main(args):
        config = args.c

        brain = Brain(config)
        brain.wake()

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, default="config/deeplodocus.yaml",
                        help="Path to the config yaml file")
    arguments = parser.parse_args()
    main(arguments)
