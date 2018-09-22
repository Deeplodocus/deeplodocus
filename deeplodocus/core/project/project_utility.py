import os.path
import os
from distutils.dir_util import copy_tree
from pathlib import Path

import __main__

from deeplodocus.utils.notification import Notification, DEEP_FATAL

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
        Generate a Deep Project
        :return: None
        """

        deeplodocus_project_path = self.main_path + "/" + self.project_name
        source_project_structre = str(Path(__file__).parent) + "/deeplodocus"

        try:
            if not os.path.exists(deeplodocus_project_path):
                os.mkdir(deeplodocus_project_path)
        except:
            Notification(DEEP_FATAL, "An error occured during the generation of the folders. Make sure the desired folder exists.")

        try:
            copy_tree(source_project_structre, deeplodocus_project_path, update= 1)
        except:
            Notification(DEEP_FATAL, "An error occurred during the copy of the files. Make sure the destination folder exists and check your Deeplodocus installation.")

    def __generate_main_path(self):
        """
        Authors: Alix Leroy
        Get the absolute path where the project deeplodocus has to be generated
        :return: Main path
        """

        path = os.path.dirname(__main__.__file__)
        return path


