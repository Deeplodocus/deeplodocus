import os
import yaml
import time

# Try and import readline
# readline makes input() behave like an interactive terminal
try:
    import readline
except ImportError:
    pass

from deeplodocus import __version__
from deeplodocus.brain.frontal_lobe import FrontalLobe
from deeplodocus.brain.thalamus import Thalamus
from deeplodocus.brain.visual_cortex import VisualCortex
from deeplodocus.utils.dict_utils import convert_dict
from deeplodocus.flags import *
from deeplodocus.utils.generic_utils import convert
from deeplodocus.utils.logo import Logo
from deeplodocus.utils.logs import Logs
from deeplodocus.utils.notification import Notification, DeepError
from deeplodocus.utils.vis_utils import plot_history
from deeplodocus.brain.visual_cortex.graph import Graph
from deeplodocus.core.project.structure.config import *


class Brain(FrontalLobe):
    """
    AUTHORS:
    --------

    :author: Alix Leroy
    :author: Samuel Westlake

    DESCRIPTION:
    ------------

    A Brain class that manages the commands of the user and allows to start the training

    PUBLIC METHODS:
    ---------------
    :method wake:
    :method sleep:
    :method save_config: Save all configuration files into self.config_dir.
    :method clear_config: Reset self.config to an empty Namespace.
    :method store_config: Store a copy of self.config in self._config.
    :method restore_config: Set self.config to a copy of self._config.
    :method load_config: Load self.config from self.config_dir.
    :method check_config: Check self.config for missing parameters and set data types accordingly.
    :method clear_logs: Deletes logs that are not to be kept, as decided in the config settings.
    :method close_logs: Closes logs that are to be kept and deletes logs that are to be deleted, see config settings.
    :method ui: Start the Visual Cortex of Deeplodocus

    PRIVATE METHODS:
    ----------------
    :method __init__:
    :method __check_config:
    :method __convert_dtype:
    :method __convert:
    :method __on_wake:
    :method __execute_command:
    :method __preprocess_command:
    :method __illegal_command_messages:
    :method __get_command_flags:
    :method __good_bye:
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
        self.__close_logs(force=True)
        Logo(version=__version__)
        FrontalLobe.__init__(self)          # Model Manager
        self.config_dir = config_dir
        self.visual_cortex = None
        time.sleep(0.5)                     # Wait for the UI to respond
        self.config = None
        self._config = None
        self.load_config()
        self.set_device()
        Thalamus()  # Initialize the Signal Manager

    """
    "
    " Public Methods
    "
    """

    def wake(self):
        """
        AUTHORS:
        --------
        :author: Samuel Westlake
        :author: Alix Leroy

        DESCRIPTION:
        ------------
        Deeplodocus terminal commands

        RETURN:
        -------
        :return: None
        """
        self.__on_wake()
        while True:
            try:
                command = Notification(DEEP_NOTIF_INPUT, DEEP_MSG_INSTRUCTRION).get()
            except KeyboardInterrupt:
                self.sleep()
            self.__execute_command(command)

    def sleep(self):
        """
        AUTHORS:
        --------
        :author: Samuel Westlake
        :author: Alix Leroy

        DESCRIPTION:
        ------------
        Stop the interface, close logs and print good-bye message

        RETURN:
        -------
        :return: None
        """
        # Stop the visual cortex
        if self.visual_cortex is not None:
            self.visual_cortex.stop()
        self.__good_bye()
        self.__close_logs()
        raise SystemExit(0)

    def save_config(self):
        """
        AUTHORS:
        --------
        :author: Samuel Westlake

        DESCRIPTION:
        ------------
        Save self.config to the config directory

        RETURN:
        -------
        :return: None
        """
        for key, namespace in self.config.get().items():
            if isinstance(namespace, Namespace):
                namespace.save("%s/%s%s" % (self.config_dir, key, DEEP_EXT_YAML))

    def clear(self, force=False):
        """
        :return:
        """
        if force:
            do_clear = True
        else:
            # Ask user if they really want to clear
            while True:
                do_clear = Notification(
                    DEEP_NOTIF_INPUT,
                    "Are you sure you want to clear this session : %s ? (yes / no)" % self.config.project.session
                ).get()
                if do_clear.lower() in ["yes", "y"]:
                    do_clear = True
                    break
                elif do_clear.lower() in ["no", "n"]:
                    do_clear = False
                    break
        if do_clear:
            Notification(DEEP_NOTIF_INFO, DEEP_MSG_BRAIN_CLEAR_ALL % self.config.project.session)
            for directory in DEEP_LOG_RESULT_DIRECTORIES:
                path = "/".join((self.config.project.session, directory))
                try:
                    for file in os.listdir(path):
                        os.remove("/".join((path, file)))
                except FileNotFoundError:
                    pass
            Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_BRAIN_CLEARED_ALL % self.config.project.session)

    def clear_history(self, force=False):
        if force:
            do_clear = True
        else:
            # Ask user if they really want to clear
            while True:
                do_clear = Notification(
                    DEEP_NOTIF_INPUT,
                    "Are you sure you want to clear history from session : %s ? (yes / no)" % self.config.project.session
                ).get()
                if do_clear.lower() in ["yes", "y"]:
                    do_clear = True
                    break
                elif do_clear.lower() in ["no", "n"]:
                    do_clear = False
                    break

        if do_clear:
            Notification(DEEP_NOTIF_INFO, DEEP_MSG_BRAIN_CLEAR_HISTORY % self.config.project.session)
            path = "/".join((self.config.project.session, "history"))
            try:
                for file in os.listdir(path):
                    os.remove("/".join((path, file)))
            except FileNotFoundError:
                pass
            Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_BRAIN_CLEARED_HISTORY % self.config.project.session)
            self.load_memory()

    def clear_logs(self, force=False):
        if force:
            do_clear = True
        else:
            # Ask user if they really want to clear
            while True:
                do_clear = Notification(
                    DEEP_NOTIF_INPUT,
                    "Are you sure you want to clear logs from session : %s ? (yes / no)" % self.config.project.session
                ).get()
                if do_clear.lower() in ["yes", "y"]:
                    do_clear = True
                    break
                elif do_clear.lower() in ["no", "n"]:
                    do_clear = False
                    break

        if do_clear:
            Notification(DEEP_NOTIF_INFO, DEEP_MSG_BRAIN_CLEAR_LOGS % self.config.project.session)
            path = "/".join((self.config.project.session, "logs"))
            try:
                for file in os.listdir(path):
                    os.remove("/".join((path, file)))
            except FileNotFoundError:
                pass
            Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_BRAIN_CLEARED_LOGS % self.config.project.session)

    def restore_config(self):
        """
        AUTHORS:
        --------
        :author: Samuel Westlake

        DESCRIPTION:
        ------------
        Restore the config to the last stored version

        RETURN:
        -------
        :return: None
        """
        self.config = self._config.copy()

    def store_config(self):
        """
        AUTHORS:
        --------
        :author: Samuel Westlake

        DESCRIPTION:
        ------------
        Saves a deep copy of the config as _config. In case the user wants to revert to previous settings

        RETURN:
        -------
        :return: None
        """
        self._config = self.config.copy()

    def load_config(self):
        """
        AUTHORS:
        --------
        :author: Samuel Westlake

        DESCRIPTION:
        ------------
        Loads each of the config files in the config directory into self.config
        self.check_config() is called penultimately
        self.store_config() is called finally

        RETURN:
        -------
        :return: None
        """
        try:
            Notification(DEEP_NOTIF_INFO, DEEP_MSG_CONFIG_LOADING_DIR % self.config_dir)
            # If the config directory exists
            if os.path.isdir(self.config_dir):
                self.config = Namespace()
                # For each expected configuration file
                for key, file_name in DEEP_CONFIG_FILES.items():
                    config_path = "%s/%s" % (self.config_dir, file_name)
                    if os.path.isfile(config_path):
                        # Notification(DEEP_NOTIF_INFO, DEEP_MSG_CONFIG_LOADING_FILE % config_path)
                        try:
                            self.config.add({key: Namespace(config_path)})
                        except yaml.scanner.ScannerError as e:
                            file, line, column = str(e).split(", ")
                            error, file = file.split("\n")
                            file = file[5:].replace('"', "")
                            Notification(
                                DEEP_NOTIF_FATAL, "Unable to load config file : %s" % error,
                                solutions="See %s : %s : %s" % (file, line, column)
                            )
                        self.check_config(key=key)
                    else:
                        self.config = Namespace()
                        Notification(DEEP_NOTIF_FATAL, DEEP_MSG_FILE_NOT_FOUND % config_path)
                Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_CONFIG_COMPLETE)
            else:
                Notification(DEEP_NOTIF_ERROR, DEEP_MSG_DIR_NOT_FOUND % self.config_dir)
            self.store_config()
        except DeepError:
            self.sleep()

    def check_config(self, key=None):
        """
        AUTHORS:
        --------
        :author: Samuel Westlake

        DESCRIPTION:
        ------------
        Checks self.config by auto-completing missing parameters with default values and converting values to the
        required data type.
        Once check_config has been run, it can be assumed that self.config is complete.
        See self.__check_config for more details.

        RETURN:
        -------
        :return: None
        """
        if key is None:
            self.__check_config()
            Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_CONFIG_COMPLETE)
        else:
            self.__check_config(DEEP_CONFIG[key], sub_space=key)

    def __clear_logs(self, force=False):
        """
        AUTHORS:
        --------
        :author: Samuel Westlake

        DESCRIPTION:
        ------------
        Deletes logs that are not to be kept, as decided in the config settings

        PARAMETERS:
        -----------
        :param force: bool Use if you want to force delete all logs

        RETURN:
        -------
        :return: None
        """
        for log_type, (directory, ext) in DEEP_LOGS.items():
            # If forced or log should not be kept, delete the log
            if force or not self.config.project.logs.get(log_type):
                Logs(log_type, directory, ext).delete()

    def __close_logs(self, force=False):
        """
        AUTHORS:
        --------
        :author: Samuel Westlake

        DESCRIPTION:
        ------------
        Closes logs that are to be kept and deletes logs that are to be deleted, as decided in the config settings

        PARAMETERS:
        -----------
        :param force: bool: Use if you want to force all logs to close (use if you don't want logs to be deleted)

        RETURN:
        -------
        :return: None
        """
        for log_type, (directory, ext) in DEEP_LOGS.items():
            # If forced to closer or log should be kept, close the log
            # NB: config does not have to exist if force is True
            if log_type == DEEP_LOG_NOTIFICATION:
                try:
                    new_directory = "/".join(
                        (get_main_path(), self.config.project.session, "logs")
                    )
                except AttributeError:
                    new_directory = None
            else:
                new_directory = None
            if force or self.config.project.logs.get(log_type):
                if os.path.isfile("%s/%s%s" % (directory, log_type, ext)):
                    Logs(log_type, directory, ext).close(new_directory)
            else:
                Logs(log_type, directory, ext).delete()

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

        if self.visual_cortex is None:
            self.visual_cortex = VisualCortex()
        else:
            Notification(DEEP_NOTIF_ERROR, "The Visual Cortex is already running.")

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

        if self.visual_cortex is not None:
            self.visual_cortex.stop()
            self.visual_cortex = None
        else:
            Notification(DEEP_NOTIF_ERROR, "The Visual Cortex is already asleep.")

    def plot_history(self, *args, **kwargs):
        """
        :param args:
        :param kwargs:
        :return:
        """
        plot_history("/".join((self.config.project.session, "history")), *args, **kwargs)

    def plot_graph(self, format="svg"):
        Graph(self.model, self.model.named_parameters())

    """
    "
    " Private Methods
    "
    """

    def __check_config(self, dictionary=DEEP_CONFIG, sub_space=None):
        """
        AUTHORS:
        --------

        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Checks the config for missing or invalid values and corrects where necessary.

        PARAMETERS:
        -----------

        :param dictionary: dictionary to check the config against.
        :param sub_space: list of strings: for internal use

        RETURN:
        -------

        :return: None
        """
        sub_space = [] if sub_space is None else sub_space
        sub_space = sub_space if isinstance(sub_space, list) else [sub_space]
        for name, value in dictionary.items():
            if isinstance(value, dict):
                keys = list(self.config.get(sub_space)) if name is DEEP_CONFIG_WILDCARD else [name]
                for key in keys:
                    if DEEP_CONFIG_DTYPE in value and DEEP_CONFIG_DEFAULT in value:
                        if self.config.check(key, sub_space=sub_space):
                            cond_1 = value[DEEP_CONFIG_DTYPE] is None
                            cond_2 = self.config.get(sub_space)[key] == value[DEEP_CONFIG_DEFAULT]
                            # If dtype is none or the item is already the default value, continue
                            if cond_1 or cond_2:
                                continue
                            else:
                                self.config.get(sub_space)[key] = self.__convert(self.config.get(sub_space)[key],
                                                                                 value[DEEP_CONFIG_DTYPE],
                                                                                 value[DEEP_CONFIG_DEFAULT],
                                                                                 sub_space=sub_space + [key])
                        else:
                            self.__add_to_config(key, value[DEEP_CONFIG_DEFAULT], sub_space)
                    else:
                        self.__check_config(dictionary=value, sub_space=sub_space + [key])

    def __add_to_config(self, key, default, sub_space):
        """
        Used by self.__config() to add a (default) value to the config
        :param key:
        :param default:
        :param sub_space:
        :return:
        """
        item_path = DEEP_CONFIG_DIVIDER.join(sub_space + [key])
        default_name = {} if isinstance(default, Namespace) else default
        Notification(DEEP_NOTIF_WARNING, DEEP_MSG_CONFIG_NOT_FOUND % (item_path, default_name))
        # self.config.get(sub_space)[key] = default
        self.config.add({key: default}, sub_space=sub_space)

    def __convert(self, value, d_type, default, sub_space=None):
        """
        AUTHORS:
        --------
        :author: Samuel Westlake

        DESCRIPTION:
        ------------
        Converts a given value to a given data type, and returns the result.
        If the value can not be converted, the given default value is returned.
        NB: If d_type is given in a list, e.g. [str], it is assume that value should be a list and each item in value
        will be converted to the given data type. If any item in the list cannot be converted, None will be returned.

        PARAMETERS:
        -----------
        :param value: the value to be converted
        :param d_type: the data type to convert the value to
        :param default: the default value to return if the current value
        :param sub_space: the list of strings that defines the current sub-space

        RETURN:
        -------
        :return: new_value
        """
        sub_space = [] if sub_space is None else sub_space
        sub_space = DEEP_CONFIG_DIVIDER.join(sub_space)

        # If the value is not set
        if value is None and default is not None:
            # Notify user that default is being used
            if default == {}:
                new_value = Namespace()
            else:
                new_value = default
                if not default == []:
                    Notification(DEEP_NOTIF_WARNING, DEEP_MSG_CONFIG_NOT_SET % (sub_space, default))
        elif d_type is dict:
            new_value = Namespace(convert_dict(value.get_all()))
        else:
            new_value = convert(value, d_type)
            if new_value is None:
                # Notify user that default is being used
                Notification(
                    DEEP_NOTIF_WARNING, DEEP_MSG_CONFIG_NOT_CONVERTED % (
                        sub_space,
                        value,
                        self.__get_dtype_name(d_type),
                        default
                    )
                )
        return new_value

    def __get_dtype_name(self, d_type):
        """
        :param d_type:
        :return:
        """
        if isinstance(d_type, list):
            return [self.__get_dtype_name(dt) for dt in d_type]
        elif isinstance(d_type, dict):
            return {key: self.__get_dtype_name(dt) for key, dt in d_type.items()}
        else:
            return d_type.__name__

    def __on_wake(self):
        """
        AUTHORS:
        --------
        :author: Samuel Westlake

        DESCRIPTION:
        ------------
        Executes the list of commands in self.config.project.on_wake

        RETURN:
        -------
        :return: None
        """
        if self.config.project.on_wake is not None:
            for command in self.config.project.on_wake:
                Notification(DEEP_NOTIF_INFO, command)
                self.__execute_command(command)

    def __execute_command(self, command):
        """
        AUTHORS:
        --------
        :author: Samuel Westlake

        DESCRIPTION:
        ------------
        Calls exec for the given command

        PARAMETERS:
        -----------
        :param command: str: the command to execute

        RETURN:
        -------
        :return: None
        """
        commands, flags = self.__preprocess_command(command)
        for command, flag in zip(commands, flags):
            if command in DEEP_EXIT_FLAGS:
                self.sleep()
            else:
                try:
                    if flag is None:
                        exec("self.%s" % command)
                    elif flag == DEEP_CMD_PRINT:
                        exec("Notification(DEEP_NOTIF_RESULT, self.%s)" % command)
                except DeepError:
                    time.sleep(0.1)
                except KeyboardInterrupt:
                    self.sleep()

    def __preprocess_command(self, command):
        """
        AUTHORS:
        --------
        :author: Samuel Westlake

        DESCRIPTION:
        ------------
        Pre-processes a given command for execuution.

        PARAMETERS:
        -----------
        :param command: str: the command to process.

        RETURN:
        -------
        :return: str: the command, str: any flags associated with the command.
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
        AUTHORS:
        --------
        :author: Samuel Westlake

        DESCRIPTION:
        ------------
        Explains why a given command is illegal.

        PARAMETERS:
        -----------
        :param command: str: the illegal command.

        RETURN:
        -------
        :return: None
        """
        message = (DEEP_MSG_ILLEGAL_COMMAND % command)
        if "__" in command or "._" in command or command.startswith("_"):
            message = "%s %s" % (message, DEEP_MSG_PRIVATE)
        if command == "wake()":
            message = "%s %s" % (message, DEEP_MSG_ALREADY_AWAKE)
        if command == "config.save":
            message = "%s %s" % (message, DEEP_MSG_USE_CONFIG_SAVE)
        Notification(DEEP_NOTIF_WARNING, message)

    @staticmethod
    def __get_command_flags(commands):
        """
        AUTHORS:
        --------
        :author: Samuel Westlake

        DESCRIPTION:
        ------------
        Extracts any flags from each command in a list of commands.

        PARAMETERS:
        -----------
        :param commands: list of str: the commands to extract flags from.

        RETURN:
        -------
        :return: None
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

    @staticmethod
    def __good_bye():
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Display thanks message

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: Universal Love <3
        """
        Notification(DEEP_NOTIF_LOVE, "=================================")
        Notification(DEEP_NOTIF_LOVE, "Thank you for using Deeplodocus !")
        Notification(DEEP_NOTIF_LOVE, "== Made by Humans with deep <3 ==")
        Notification(DEEP_NOTIF_LOVE, "=================================")

    """
    "
    " Aliases
    "
    """

    visual_cortex = ui
    vc = ui
    user_interface = ui


brain = None

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
