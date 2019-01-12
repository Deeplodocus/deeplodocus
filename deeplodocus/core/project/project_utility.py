import os
from distutils.dir_util import copy_tree

from deeplodocus.utils.namespace import Namespace
from deeplodocus.utils.notification import Notification
from deeplodocus.utils.flags.config import DEEP_CONFIG
from deeplodocus.utils.flags.ext import DEEP_EXT_YAML
from deeplodocus.utils.flags.msg import *
from deeplodocus.utils.flags.notif import *
from deeplodocus.utils import get_main_path


class ProjectUtility(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Generate a project deeplodocus ready to use
    """

    def __init__(self, project_name="deeplodocus_project", main_path=None, force_overwrite=False):
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

        # Get the path to the original files
        source_project_structure = "%s/deep_structure" % os.path.abspath(os.path.dirname(__file__))

        # Check if the project already exists
        if self.__check_exists(project_path):

            # If we can continue copy the deep structure from original folder to the deep project folder
            copy_tree(source_project_structure, project_path, update=1)

            # Copy the required config files
            # self.__init_config()

            # Clean the structure (remove __pycache__ folder and __ini__.py files)
            self.__clean_structure(project_path)
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
        config = self.__set_config_defaults(config)
        config_dir = "%s/%s/config" % (self.main_path, self.project_name)
        os.makedirs(config_dir, exist_ok=True)
        for key, namespace in config.get().items():
            if isinstance(namespace, Namespace):
                namespace.save("%s/%s%s" % (config_dir, key, DEEP_EXT_YAML))

    def __set_config_defaults(self, namespace):
        """
        :param namespace:
        :return:
        """
        for key, item in namespace.get().items():
            if isinstance(item, Namespace):
                if "default" in item.get():
                    namespace.get()[key] = item.get()["default"]
                else:
                    item = self.__set_config_defaults(item)
        return namespace

    @staticmethod
    def __clean_structure(deeplodocus_project_path):
        """
        AUTHORS:
        --------

        :author: Alix Leroy and SW

        DESCRIPTION:
        ------------

        Remove __init__.py files and __pycache__ folders

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        # Browse inside all files and folders in the new generated project
        for root, dirs, files in os.walk(deeplodocus_project_path):
            # Remove init files
            if os.path.isfile(root + "/__init__.py"):
                os.remove(root + "/__init__.py")
            # Remove pycache folders
            [dirs.remove(dirname) for dirname in dirs[:] if dirname.startswith('.') or dirname == '__pycache__']
