import os.path
import time
import random

from deeplodocus.utils.notification import Notification
from deeplodocus.utils.flags import *
from deeplodocus.utils.namespace import Namespace
from deeplodocus.utils.logo import Logo
from deeplodocus.utils.end import End
from deeplodocus.utils.logs import Logs
from deeplodocus.ui.user_interface import UserInterface
from deeplodocus import __version__
import __main__


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

    def __init__(self, config_dir):
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

        self.write_logs = True   # Initially set to True and updated after the config is loaded
        self.config_dir = config_dir
        self.logs = [["notification", DEEP_PATH_NOTIFICATION, ".logs"],
                     ["history_train_batches", DEEP_PATH_HISTORY, ".csv"],
                     ["history_train_epochs", DEEP_PATH_HISTORY, ".csv"],
                     ["history_validation", DEEP_PATH_HISTORY, ".csv"]]
        self.__init_logs()
        Logo(version=__version__, write_logs=self.write_logs)
        self.config = Namespace()
        self.user_interface = None
        time.sleep(0.5)
        self.load_config()

    def wake(self):
        """
        Authors : Alix Leroy, SW
        Main of deeplodocus framework
        :return: None
        """
        sleep = False
        while not sleep:
            command = Notification(DEEP_NOTIF_INPUT, DEEP_MSG_INSTRUCTRION, write_logs=self.write_logs).get()
            for command in self.__preprocess_command(command):
                if command in DEEP_EXIT_FLAGS:
                    sleep = True
                    break
                else:
                    exec("self.%s" % command)
                    time.sleep(0.5)
        if self.user_interface is not None:
            self.user_interface.stop()
        End(error=False)

    def covfefe(self):
        """
        :return:
        """
        garbage = ["These results will be the best results in history... Very clever!",
                   "I know parameters, I have the best parameters."]
        Notification(DEEP_NOTIF_INFO, random.choice(garbage),
                     write_logs=False)

    def __preprocess_command(self, command):
        """
        Author: SW
        :param command: str: raw command input from user
        :return: list of str: split and filtered commands for sequential execution
        """
        command = command.replace(" ", "")
        commands = command.split("&&")
        for command in commands:
            remove = False
            for prefix in DEEP_FILTER_STARTS_WITH:
                if command.startswith(prefix):
                    remove = True
                    break
            if not remove:
                for suffix in DEEP_FILTER_ENDS_WITH:
                    if command.endswith(suffix):
                        remove = True
            if not remove:
                for item in DEEP_FILTER:
                    if command == item:
                        remove = True
            if not remove:
                for item in DEEP_FILTER_STARTS_ENDS_WITH:
                    if command.startswith(item) and command.endswith(item):
                        remove = True
            if not remove:
                for item in DEEP_FILTER_INCLUDES:
                    if item in command:
                        remove = True
            if remove:
                commands.remove(command)
                message = (DEEP_MSG_REMOVE_COMMAND % command)
                if command == "wake":
                    message = "%s, %s" % (message, DEEP_MSG_ALREADY_AWAKE)
                Notification(DEEP_NOTIF_WARNING, message, write_logs=self.write_logs)
        return commands

    def load_config(self):
        """
        Author: SW
        Function: Checks current config path is valid, if not, user is prompted to give another
        :return: bool: True if a valid config path is set, otherwise, False
        """
        Notification(DEEP_NOTIF_INFO, DEEP_MSG_LOAD_CONFIG_START % self.config_dir, write_logs=self.write_logs)
        # If the config directory exists
        if os.path.isdir(self.config_dir):
            self.config = Namespace()
            # For each expected configuration file
            for key, file_name in DEEP_CONFIG_FILES.items():
                config_path = "%s/%s" % (self.config_dir, file_name)
                # If the expected file exists
                if os.path.isfile(config_path):
                    self.config.add({key: Namespace(config_path)})
                    Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_LOAD_CONFIG_FILE % config_path, write_logs=self.write_logs)
                else:
                    Notification(DEEP_NOTIF_WARNING, DEEP_MSG_FILE_NOT_FOUND % config_path, write_logs=self.write_logs)
            if self.check_config():
                Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_LOAD_CONFIG_SUCCESS % self.config_dir, write_logs=self.write_logs)
            else:
                Notification(DEEP_NOTIF_ERROR, DEEP_MSG_LOAD_CONFIG_FAIL, write_logs=self.write_logs)
        else:
            Notification(DEEP_NOTIF_ERROR, DEEP_MSG_DIR_NOT_FOUND % self.config_dir, write_logs=self.write_logs)
        self.__set_write_logs()

    def check_config(self, dictionary=DEEP_CONFIG, sub_space=None):
        """
        Author: SW
        :return: bool: whether the config has every expected entry or not
        """
        complete = True
        sub_space = [] if sub_space is None else sub_space
        sub_space = sub_space if isinstance(sub_space, list) else [sub_space]
        for key, items in dictionary.items():
            this_sub_sapce = sub_space + [key]
            for item in items:
                if isinstance(item, dict):
                    if not self.check_config(item, this_sub_sapce):
                        complete = False
                else:
                    try:
                        exists = self.config.check(item, this_sub_sapce)
                    except (AttributeError, KeyError):
                        exists = False
                    if not exists:
                        complete = False
                        item_path = DEEP_CONFIG_DIVIDER.join(this_sub_sapce[1:] + [item])
                        file = this_sub_sapce[0]
                        Notification(DEEP_NOTIF_WARNING, DEEP_MSG_CONFIG_NOT_FOUND % (item_path, file),
                                     write_logs=self.write_logs)
        return complete

    def __set_write_logs(self):
        """
        Author: SW
        Sets self.write logs if the project configurations have been written
        :return: None
        """
        if self.config.check("write_logs", DEEP_CONFIG_PROJECT):
            self.write_logs = self.config.project.write_logs
            if self.write_logs is False:
                Logs(type="notification",
                     extension=DEEP_EXT_LOGS,
                     folder="%s/logs" % os.path.dirname(os.path.abspath(__main__.__file__))).delete()
                Notification(DEEP_NOTIF_INFO, DEEP_MSG_REMOVE_LOGS, write_logs=False)

    def __run_command(self, command: str) -> None:
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
        for log_name, log_folder, log_extension in self.logs:
            Logs(log_name, log_folder, log_extension).check_init()
        Notification(DEEP_NOTIF_SUCCESS, "Log and History files initialized ! ", write_logs=self.write_logs)


if __name__ == "__main__":
    import argparse

    def main(args):
        config_dir = args.c
        brain = Brain(config_dir)
        brain.wake()

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, default="core/project/deep_structure/config",
                        help="Path to the config directory")
    arguments = parser.parse_args()
    main(arguments)
