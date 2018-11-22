import os
import time

from deeplodocus.utils.notification import Notification
from deeplodocus.utils.flags import *
from deeplodocus.utils.namespace import Namespace
from deeplodocus.utils.logo import Logo
from deeplodocus.utils.end import End
from deeplodocus.brain.frontal_lobe import FrontalLobe
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

        RETURN:
        -------

        :return: None
        """
        self.close_logs(force=True)
        self.config_dir = config_dir
        self.config_is_complete = False
        Logo(version=__version__)
        self.user_interface = None
        time.sleep(0.5)                     # Wait for the UI to respond
        self.frontal_lobe = None
        self.config = None
        self._config = None
        self.load_config()

    def wake(self):
        """
        Authors : Alix Leroy, SW
        Deeplodocus terminal commands
        :return: None
        """
        self.frontal_lobe = FrontalLobe(self.config)
        self.__on_wake()
        while True:
            command = Notification(DEEP_NOTIF_INPUT, DEEP_MSG_INSTRUCTRION).get()
            self.__execute_command(command)

    def sleep(self):
        """
        Author: SW, Alix Leroy
        Stop the interface, close logs and print good-bye message
        :return: None
        """
        if self.user_interface is not None:
            self.user_interface.stop()
        self.close_logs()
        End(error=False)

    def save_config(self):
        """
        Author: SW
        Save the config to the config folder
        :return: None
        """
        for key, namespace in self.config.get().items():
            if isinstance(namespace, Namespace):
                namespace.save("%s/%s%s" % (self.config_dir, key, DEEP_EXT_YAML))

    def clear_config(self):
        """
        Author: SW
        Reset the config to an empty Namespace
        :return: None
        """
        self.config = Namespace()

    def restore_config(self):
        """
        Author: SW
        Restore the config to the last stored version
        :return:
        """
        self.config = self._config.copy()

    def store_config(self):
        """
        Author: SW
        Saves a deep copy of the config as _config. In case the user wants to revert to previous settings
        :return: None
        """
        self._config = self.config.copy()

    def load_config(self):
        """
        Author: SW
        Function: Checks current config path is valid, if not, user is prompted to give another
        :return: bool: True if a valid config path is set, otherwise, False
        """
        Notification(DEEP_NOTIF_INFO, DEEP_MSG_LOAD_CONFIG_START % self.config_dir)
        # If the config directory exists
        if os.path.isdir(self.config_dir):
            self.clear_config()
            # For each expected configuration file
            for key, file_name in DEEP_CONFIG_FILES.items():
                config_path = "%s/%s" % (self.config_dir, file_name)
                if os.path.isfile(config_path):
                    self.config.add({key: Namespace(config_path)})
                    Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_LOAD_CONFIG_FILE % config_path)
                else:
                    Notification(DEEP_NOTIF_ERROR, DEEP_MSG_FILE_NOT_FOUND % config_path)
            self.check_config()
            self.store_config()
        else:
            Notification(DEEP_NOTIF_ERROR, DEEP_MSG_DIR_NOT_FOUND % self.config_dir)

    def clear_logs(self, force=False):
        """
        Author: SW
        Deletes logs that are not to be kept, as decided in the config settings
        :param force: bool Use if you want to force delete all logs
        :return: None
        """
        for log_type, (directory, ext) in DEEP_LOGS.items():
            # If forced or log should not be kept, delete the log
            if force or not self.config.project.logs.get(log_type):
                Logs(log_type, directory, ext).delete()

    def close_logs(self, force=False):
        """
        Author: SW
        Closes logs that are to be kept and deletes logs that are to be deleted, as decided in the config settings
        :param force: bool: Use if you want to force all logs to close (use if you don't want logs to be deleted)
        :return: None
        """
        for log_type, (directory, ext) in DEEP_LOGS.items():
            # If forced to closer or log should be kept, close the log
            # NB: config does not have to exist if force is True
            if force or self.config.project.logs.get(log_type):
                if os.path.isfile("%s/%s%s" % (directory, log_type, ext)):
                    Logs(log_type, directory, ext).close()
            else:
                Logs(log_type, directory, ext).delete()

    def check_config(self):
        """
        Author: SW
        Check the contents of the config namespace against the expecting contents listed in DEEP_CONFIG
        :return: bool: True if config is complete, False otherwise
        """
        if self.__check_config():
            Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_LOAD_CONFIG_SUCCESS)
            return True
        else:
            return False

    def train(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Train the model is the model is loaded

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """

        if self.frontal_lobe is not None:
            self.frontal_lobe.train()
        else:
            Notification(DEEP_NOTIF_ERROR, "The model is not loaded yet, please feed Deeplodocus with all the required config files.")

    def load(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Load the content in the Frontal Lobe

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """

        if self.frontal_lobe is not None:
            self.frontal_lobe.load()
        else:
            Notification(DEEP_NOTIF_ERROR, "The model is not loaded yet, please feed Deeplodocus with all the required config files.")

    def load_model(self):
        """
        :return:
        """
        self.frontal_lobe.load_model()

    def load_optimizer(self):
        """
        :return:
        """
        self.frontal_lobe.load_optimizer()
        
    def ui(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Start the User Interface

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        if self.user_interface is None:
            self.user_interface = UserInterface()
        else:
            Notification(DEEP_NOTIF_ERROR, "The User Interface is already running.")


    def stop_ui(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Stop the User Interface

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """

        if self.user_interface is not None:
            self.user_interface.stop()
        else:
            Notification(DEEP_NOTIF_ERROR, "The User Interface is not currently running.")

    def __on_wake(self):
        """
        Author: SW
        Execute any commands listed under config/project/on_wake
        :return: None
        """
        if self.config.project.on_wake is not None:
            if not isinstance(self.config.project.on_wake, list):
                self.config.project.on_wake = [self.config.project.on_wake]
            for command in self.config.project.on_wake:
                self.__execute_command(command)

    def __execute_command(self, command):
        """
        Author: SW
        :param command: str: the command to be executed
        :return: None
        """
        commands, flags = self.__preprocess_command(command)
        for command, flag in zip(commands, flags):
            if command in DEEP_EXIT_FLAGS:
                self.sleep()
            else:
                #try:
                if flag is None:
                    exec("self.%s" % command)
                elif flag == DEEP_CMD_PRINT:
                    exec("Notification(DEEP_NOTIF_RESULT, self.%s)" % command)
                #except AttributeError as e:
                    #Notification(DEEP_NOTIF_ERROR, str(e), **self.config.project.notifications.get())
                time.sleep(0.5)

    def __preprocess_command(self, command):
        """
        Author: SW
        :param command: str: raw command input from user
        :return: list of str: split and filtered commands for sequential execution
        """
        commands = command.split(" & ")
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
                self.__illegal_command_messages(command)
        return self.__get_command_flags(commands)

    @staticmethod
    def __illegal_command_messages(command):
        """
        Author: SW
        Display reasons why a given command is illegal
        :param command: str: command to be executed
        :return:
        """
        message = (DEEP_MSG_ILLEGAL_COMMAND % command)
        if "__" in command or "._" in command or command.startswith("_"):
            message = "%s %s" % (message, DEEP_MSG_PRIVATE)
        if command == "wake()":
            message = "%s %s" % (message, DEEP_MSG_ALREADY_AWAKE)
        if command == "config.save":
            message = "%s %s" % (message, DEEP_MSG_USE_CONFIG_SAVE)
        Notification(DEEP_NOTIF_WARNING, message)

    def __check_config(self, dictionary=DEEP_CONFIG, sub_space=None):
        """
        Author: SW
        :return: bool: whether the config has every expected entries or not
        """
        complete = True
        sub_space = [] if sub_space is None else sub_space
        sub_space = sub_space if isinstance(sub_space, list) else [sub_space]
        for key, value in dictionary.items():
            if isinstance(value, dict):
                if not self.__check_config(value, sub_space + [key]):
                    complete = False
            else:
                if not self.config.check(key, sub_space):
                    complete = False
                    item_path = DEEP_CONFIG_DIVIDER.join(sub_space + [key])
                    Notification(DEEP_NOTIF_ERROR, DEEP_MSG_CONFIG_NOT_FOUND % item_path)
        return complete

    @staticmethod
    def __get_command_flags(commands):
        """
        Author: SW
        For separating any flags from given commands
        :param commands: list of str: commands to be executed
        :return: commands with any given flags
        """
        flags = [None for _ in range(len(commands))]
        for i, command in enumerate(commands):
            for flag in DEEP_CMD_FLAGS:
                if flag in command:
                    flags[i] = flag
                    commands[i] = command.replace(" %s" % flag, "")
                else:
                    flags[i] = None
        return commands, flags


if __name__ == "__main__":
    import argparse

    def main(args):
        config_dir = args.c
        brain = Brain(config_dir)
        brain.wake()

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, default="config",
                        help="Path to the config directory")
    arguments = parser.parse_args()
    main(arguments)
