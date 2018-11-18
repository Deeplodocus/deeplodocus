import os.path
import os
from distutils.dir_util import copy_tree

from deeplodocus.utils.notification import Notification
from deeplodocus.utils.flags import *

class ProjectUtility(object):
    """
    Authors : Alix Leroy,
    Generate a project deeplodocus ready to use
    """

    def __init__(self, project_name = "deeplodocus_project", main_path = None):
        """
        Authors: Alix Leroy,
        """
        self.project_name = project_name
        if main_path is None:
            self.main_path = self.__generate_main_path()


    def get_main_path(self):
        """
        Authors: Alix Leroy,
        Getter for the attribute : main_path
        :return: main_path
        """
        return self.main_path


    def generate_structure(self):
        """
        Authors : Alix Leroy,
        Generate a Deep Project by copying the default folder to the desired location
        :return: None
        """

        deeplodocus_project_path = self.main_path + "/" + self.project_name
        source_project_structure = os.path.abspath(os.path.dirname(__file__)) + "/deep_structure"

        try:
            if not os.path.exists(deeplodocus_project_path):
                os.mkdir(deeplodocus_project_path)
        except:
            Notification(DEEP_NOTIF_FATAL, "An error occured during the generation of the folders. Make sure the desired folder exists.", log=False)

        try:
            # Copy the whole structure of a Deeplodocus project
            copy_tree(source_project_structure, deeplodocus_project_path, update= 1)
        except:
            Notification(DEEP_NOTIF_FATAL, "An error occurred during the copy of the files. Make sure the destination folder exists and check your Deeplodocus installation.", log=False)

        self.__clean_structure(deeplodocus_project_path)

        Notification(DEEP_NOTIF_SUCCESS, "Project successfully generated ! Have fun <3 ", log=False)

    def __clean_structure(self, deeplodocus_project_path):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

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

        for root, dirs, files in os.walk(deeplodocus_project_path):

            # Remove init files
            for filename in files[:]:
                if filename == "__init__.py":
                    os.remove(root +"/__init__.py")

            # Remove pycache folders
            for dirname in dirs[:]:
                if dirname.startswith('.') or dirname == '__pycache__':
                    dirs.remove(dirname)

    def __generate_main_path(self):
        """
        Authors: Alix Leroy
        Get the absolute path where the project deeplodocus has to be generated
        :return: Main path
        """

        path = os.getcwd()
        return path


