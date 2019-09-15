import os
import shutil
from typing import Optional

from deeplodocus.utils.namespace import Namespace
from deeplodocus.utils.notification import Notification
from deeplodocus.utils import get_main_path
from deeplodocus.flags.msg import *
from deeplodocus.flags.notif import *
from deeplodocus.core.project.structure.config import *
from deeplodocus.core.project.structure import DEEP_DIRECTORY_TREE



class ProjectUtility(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Generate a project deeplodocus ready to use
    """

    def __init__(self, project_name="deeplodocus_project", main_path: Optional[str] = None, force_overwrite: bool = False):
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Initialize a instance ready to create a new Deeplodocus project

        PARAMETERS:
        -----------

        :param project_name(str, optional): The name of the project
        :param main_path(str, optional): The path of the working directory
        :param force_overwrite(bool, optional): Whether we want to overwrite an existing project or not

        RETURN:
        -------

        None
        """
        self.project_name = project_name
        self.force_overwrite = force_overwrite

        if main_path is None:
            self.main_path = get_main_path()
        else:
            self.main_path = main_path

    def generate_structure(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Generate a Deep Project by copying the default folder to the desired location

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        # Path in which the deep structure will be copied
        project_path = "%s/%s" % (self.main_path, self.project_name)
        # Check if the project already exists
        if self.__check_exists(project_path):
            # Initialise the config files from config.py
            self.__init_config()
            # Initialise other project directories and files from DEEP_DIRECTORY_TREE
            self.__init_directory_tree(
                DEEP_DIRECTORY_TREE,
                file_source="%s/structure" % os.path.abspath(os.path.dirname(__file__))
            )
            Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_PROJECT_GENERATED, log=False)
        else:
            Notification(DEEP_NOTIF_INFO, DEEP_MSG_PROJECT_NOT_GENERATED, log=False)

    def __check_exists(self, project_path):
        """
        Author: SW
        Checks that the project does not exist already
        If the project exists already, the user is asked if the existing project should be overwritten
        If the project does not exist already, the project directory is made
        :param project_path: str: absolute path to the main project directory
        :return: bool: whether or not project generation should continue
        """
        if os.path.isdir(project_path) and not self.force_overwrite:
            while True:
                Notification(DEEP_NOTIF_WARNING, DEEP_MSG_PROJECT_ALREADY_EXISTS % project_path, log=False)
                proceed = Notification(DEEP_NOTIF_INPUT, DEEP_MSG_CONTINUE, log=False).get()
                if proceed.lower() in ["y", "yes"]:
                    return True
                elif proceed.lower() in ["n", "no"]:
                    return False
        else:
            os.makedirs(project_path, exist_ok=True)
            return True

    def __init_config(self):
        """
        Author: SW
        Writes the config directory and files into the project directory using DEEP_CONFIG
        :return: None
        """
        config = Namespace(DEEP_CONFIG)
        self.__rename_wildcards(config)
        self.__set_config_defaults(config)
        config_dir = "%s/%s/config" % (self.main_path, self.project_name)
        os.makedirs(config_dir, exist_ok=True)
        for key, namespace in config.get().items():
            if isinstance(namespace, Namespace):
                namespace.save("%s/%s%s" % (config_dir, key, DEEP_EXT_YAML))

    def __init_directory_tree(self, dictionary, file_source=".", parent=None):
        if parent is None:
            parent = ["."]
        for key, item in dictionary.items():
            if key == "FILES":
                for file_path in item:
                    shutil.copy(
                        src="/".join((file_source, file_path)),
                        dst="/".join([self.main_path, self.project_name] + parent)
                    )
            else:
                os.makedirs(
                    "/".join(
                        [self.main_path, self.project_name]
                        + parent
                        + [key]
                    ),
                    exist_ok=True
                )
            if isinstance(item, dict):
                self.__init_directory_tree(
                    item,
                    file_source=file_source,
                    parent=parent+[key]
                )

    def __rename_wildcards(self, namespace):
        """
        :param namespace:
        :return:
        """
        for key, item in namespace.get().items():
            if isinstance(item, Namespace):
                if DEEP_CONFIG_WILDCARD in item.get():
                    item.rename(DEEP_CONFIG_WILDCARD, DEEP_CONFIG_WILDCARD_DEFAULT[key])
                else:
                    self.__rename_wildcards(item)

    def __set_config_defaults(self, namespace):
        """
        :param namespace:
        :return:
        """
        for key, item in namespace.get().items():
            if isinstance(item, Namespace):
                if DEEP_CONFIG_DEFAULT in item.get() and DEEP_CONFIG_DTYPE in item.get():
                    try:
                        namespace.get()[key] = item.get()[DEEP_CONFIG_INIT]
                    except KeyError:
                        namespace.get()[key] = item.get()[DEEP_CONFIG_DEFAULT]
                else:
                    self.__set_config_defaults(item)
            elif isinstance(item, list):
                for i in item:
                    if isinstance(i, Namespace):
                        self.__set_config_defaults(i)
