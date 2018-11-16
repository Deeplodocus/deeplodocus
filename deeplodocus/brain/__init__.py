import os.path
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
        :param write_logs->bool: Whether to write logs or not

        RETURN:
        -------

        :return: None
        """
        self.write_logs = True   # Initially set to True and updated after the config is loaded
        self.config_dir = config_dir
        self.config_is_complete = False
        self.logs = [["notification", DEEP_PATH_NOTIFICATION, DEEP_EXT_LOGS],
                     ["history_train_batches", DEEP_PATH_HISTORY, DEEP_EXT_CSV],
                     ["history_train_epochs", DEEP_PATH_HISTORY, DEEP_EXT_CSV],
                     ["history_validation", DEEP_PATH_HISTORY, DEEP_EXT_CSV]]
        self.__init_logs()
        Logo(version=__version__, write_logs=self.write_logs)
        self.user_interface = None
        time.sleep(0.5)         # Wait for the UI to respond
        self.frontal_lobe = None
        self.config = None
        self._config = None
        self.load_config()

    def wake(self):
        """
        Authors : Alix Leroy, SW
        Main of deeplodocus framework
        :return: None
        """
        sleep = self.__on_wake()
        while not sleep:
            command = Notification(DEEP_NOTIF_INPUT, DEEP_MSG_INSTRUCTRION, write_logs=self.write_logs).get()
            sleep = self.__execute_command(command)
        self.sleep()

    def sleep(self):
        """
        :return:
        """
        if self.user_interface is not None:
            self.user_interface.stop()
        End(error=False)

    def save_config(self):
        for key, namespace in self.config.get().items():
            if isinstance(namespace, Namespace):
                namespace.save("%s/%s%s" % (self.config_dir, key, DEEP_EXT_YAML))

    def clear_config(self):
        """
        :return:
        """
        self.config = Namespace()

    def restore_config(self):
        """
        :return:
        """
        self.config = self._config.copy()

    def store_config(self):
        """
        :return:
        """
        self._config = self.config.copy()

    def load_config(self):
        """
        Author: SW
        Function: Checks current config path is valid, if not, user is prompted to give another
        :return: bool: True if a valid config path is set, otherwise, False
        """
        Notification(DEEP_NOTIF_INFO, DEEP_MSG_LOAD_CONFIG_START % self.config_dir, write_logs=self.write_logs)
        # If the config directory exists
        if os.path.isdir(self.config_dir):
            self.clear_config()
            # For each expected configuration file
            for key, file_name in DEEP_CONFIG_FILES.items():
                config_path = "%s/%s" % (self.config_dir, file_name)
                if os.path.isfile(config_path):
                    self.config.add({key: Namespace(config_path)})
                    Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_LOAD_CONFIG_FILE % config_path,
                                 write_logs=self.write_logs)
                else:
                    Notification(DEEP_NOTIF_ERROR, DEEP_MSG_FILE_NOT_FOUND % config_path, write_logs=self.write_logs)
            self.check_config()
            self.store_config()
        else:
            Notification(DEEP_NOTIF_ERROR, DEEP_MSG_DIR_NOT_FOUND % self.config_dir, write_logs=self.write_logs)
        self.__set_write_logs()

    def check_config(self):
        """
        :return:
        """
        if self.__check_config():
            Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_LOAD_CONFIG_SUCCESS, write_logs=self.write_logs)
            self.config_is_complete = True
            self.frontal_lobe = FrontalLobe(self.config, write_logs=self.write_logs)
        else:
            Notification(DEEP_NOTIF_ERROR, DEEP_MSG_LOAD_CONFIG_FAIL, write_logs=self.write_logs)
            self.config_is_complete = False
        return self.config_is_complete


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
            Notification(DEEP_NOTIF_ERROR, "The model is not loaded yet, please feed Deeplodocus with all the required config files.", write_logs=self.write_logs)

    def __on_wake(self):
        """
        :return:
        """
        if self.config.project.on_wake is not None:
            if not isinstance(self.config.project.on_wake, list):
                self.config.project.on_wake = [self.config.project.on_wake]
            for command in self.config.project.on_wake:
                if self.__execute_command(command):
                    return True
        return False

    def __execute_command(self, command):
        """
        :param command:
        :return:
        """
        commands, flags = self.__preprocess_command(command)
        for command, flag in zip(commands, flags):
            if command in DEEP_EXIT_FLAGS:
                return True
            else:
                try:
                    if flag is None:
                        exec("self.%s" % command)
                    elif flag == DEEP_CMD_PRINT:
                        exec("Notification(DEEP_NOTIF_RESULT, self.%s, write_logs=self.write_logs)" % command)
                except AttributeError as e:
                    Notification(DEEP_NOTIF_ERROR, str(e), write_logs=self.write_logs)
                time.sleep(0.5)
        return False

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

    def __illegal_command_messages(self, command):
        """
        :param command:
        :return:
        """
        message = (DEEP_MSG_ILLEGAL_COMMAND % command)
        if command == "wake()":
            message = "%s %s" % (message, DEEP_MSG_ALREADY_AWAKE)
        if command == "config.save":
            message = "%s %s" % (message, DEEP_MSG_USE_CONFIG_SAVE)
        Notification(DEEP_NOTIF_WARNING, message, write_logs=self.write_logs)

    def __check_config(self, dictionary=DEEP_CONFIG, sub_space=None):
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
                    if not self.__check_config(item, this_sub_sapce):
                        complete = False
                else:
                    try:
                        exists = self.config.check(item, this_sub_sapce)
                    except (AttributeError, KeyError):
                        exists = False
                    if not exists:
                        complete = False
                        item_path = DEEP_CONFIG_DIVIDER.join(this_sub_sapce + [item])
                        Notification(DEEP_NOTIF_ERROR, DEEP_MSG_CONFIG_NOT_FOUND % item_path,
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
                try:
                    Logs(type="notification",
                         extension=DEEP_EXT_LOGS,
                         folder="%s/logs" % os.path.dirname(os.path.abspath(__main__.__file__))).delete()
                    Notification(DEEP_NOTIF_INFO, DEEP_MSG_REMOVE_LOGS, write_logs=False)
                except FileNotFoundError:
                    pass

    def __init_logs(self):
        """
        Authors : Alix Leroy
        Initialize all logs
        :return:None
        """
        for log_name, log_folder, log_extension in self.logs:
            Logs(log_name, log_folder, log_extension).check_init()
        Notification(DEEP_NOTIF_SUCCESS, "Log and History files initialized ! ", write_logs=self.write_logs)

    @staticmethod
    def __get_command_flags(commands):
        """
        :param commands:
        :return:
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
    parser.add_argument("-c", type=str, default="../core/project/deep_structure/config",
                        help="Path to the config directory")
    arguments = parser.parse_args()
    main(arguments)
