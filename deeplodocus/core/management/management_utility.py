# Python imports
import sys
import os
import datetime
import time

# Deeplodocus imports
from deeplodocus.utils.notification import Notification
from deeplodocus.core.project.project_utility import ProjectUtility
from deeplodocus import __version__
from deeplodocus.flags.admin import *
from deeplodocus.flags.notif import *
from deeplodocus.core.project.generators import *
from deeplodocus.core.project.structure.transformers import *
from deeplodocus.brain import Brain

# Deeplodocus flags
from deeplodocus.flags import DEEP_LIST_ADMIN


class ManagementUtility(object):

    def __init__(self, argv=None):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize the ManagementUtility and keep the args in memory

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """

        self.argv = argv or sys.argv[:]

    def execute_from_command_line(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Execute the command entered in the terminal

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """

        if len(self.argv) > 1:
            if DEEP_ADMIN_NEW_PROJECT.corresponds(str(self.argv[1])):
                self.__newproject(main_path=os.getcwd())
            elif DEEP_ADMIN_VERSION.corresponds(str(self.argv[1])):
                self.__version()
            elif DEEP_ADMIN_HELP.corresponds(str(self.argv[1])):
                self.__help()
            elif DEEP_ADMIN_RUN.corresponds(str(self.argv[1])):
                self.__run_project()
            elif DEEP_ADMIN_OUTPUT_TRANSFORMER.corresponds((str(self.argv[1]))):
                self.__output_transformer()
            elif DEEP_ADMIN_ONEOF_TRANSFORMER.corresponds((str(self.argv[1]))):
                self.__oneof_transformer()
            elif DEEP_ADMIN_SEQUENTIAL_TRANSFORMER.corresponds((str(self.argv[1]))):
                self.__sequential_transformer()
            elif DEEP_ADMIN_TRANSFORMER.corresponds((str(self.argv[1]))):
                self.__transformer()
            else:
                Notification(
                    DEEP_NOTIF_ERROR,
                    "The following command does not exist : %s" % str(self.argv[1]),
                    log=False
                )
        else:
            self.__help()

    def __transformer(self):
        options = {
            1: {
                "name": "Sequential transformer",
                "method": self.__sequential_transformer
            },
            2: {
                "name": "One-of transformer",
                "method": self.__oneof_transformer
            },
            3: {
                "name": "Some-of transformer",
                "method": self.__someof_transformer
            },
            4: {
                "name": "Output transformer",
                "method": self.__output_transformer
            },
            5: {
                "name": "Cancel",
                "method": None
            }
        }
        selection = Notification(
            DEEP_NOTIF_INPUT,
            """Which transformer would you like to create?
Select one of:
\t%s""" % "\n\t".join(["%s: %s" % (key, item["name"]) for key, item in options.items()]),
            log=False).get()
        try:
            method = options[int(selection)]["method"]
        except (KeyError, ValueError):
            method = None
            Notification(DEEP_NOTIF_ERROR, "Sorry, your selection was not recognized", log=False)
        if method is not None:
            method()

    def __output_transformer(self):
        try:
            filename = self.argv[2]
        except IndexError:
            filename = "output_transformer.yaml"
        filename = self.__generate_filename(filename)
        generate_transformer(OUTPUT_TRANSFORMER, filename=filename)
        Notification(DEEP_NOTIF_SUCCESS, "New output transformer file created : %s" % filename, log=False)

    def __oneof_transformer(self):
        try:
            filename = self.argv[2]
        except IndexError:
            filename = "oneof_transformer.yaml"
        filename = self.__generate_filename(filename)
        generate_transformer(ONEOF_TRANSFORMER, filename=filename)
        Notification(DEEP_NOTIF_SUCCESS, "New one-of transformer file created : %s" % filename, log=False)

    def __sequential_transformer(self):
        try:
            filename = self.argv[2]
        except IndexError:
            filename = "sequential_transformer.yaml"
        filename = self.__generate_filename(filename)
        generate_transformer(SEQUENTIAL_TRANSFORMER, filename=filename)
        Notification(DEEP_NOTIF_SUCCESS, "New sequential transformer file created : %s" % filename, log=False)

    def __someof_transformer(self):
        try:
            filename = self.argv[2]
        except IndexError:
            filename = "someof_transformer.yaml"
        filename = self.__generate_filename(filename)
        generate_transformer(SOMEOF_TRANSFORMER, filename=filename)
        Notification(DEEP_NOTIF_SUCCESS, "New some-of transformer file created : %s" % filename, log=False)

    def __run_project(self):
        try:
            config_dir = self.argv[2]
        except IndexError:
            config_dir = "./config"
        brain = Brain(config_dir=config_dir)
        brain.wake()

    @staticmethod
    def __generate_filename(filename, n=2):
        """
        Checks that a file with the given name doesn't already exist
        If a file with the same name exists, adds (n) to the end of the name
        E.g. if "example.txt" already exists, "example(2).txt will be returned
        :param filename: str: the desired file name
        :param n: int: the starting number
        :return: str: a file name that doesn't already exist
        """
        new_filename = filename
        while True:
            files = os.listdir(".")
            if new_filename in files:
                name, ext = filename.split(".")
                new_filename = "%s(%i).%s" % (name, n, ext)
                n += 1
            else:
                break
        return new_filename

    @staticmethod
    def __help():
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Display the command available
        The commands are listed into the DEEP_LIST_ADMIN flag

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        for flag in DEEP_LIST_ADMIN:
            Notification(DEEP_NOTIF_INFO, str(flag.get_description()), log=False)

    @staticmethod
    def __version():
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Display the installed version of Deeplodocus

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        version = str(__version__)
        Notification(DEEP_NOTIF_INFO, "DEEPLODOCUS VERSION : " + str(version),  log=False)

    def __newproject(self, main_path: str):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Generate a Deeplodocus project

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """

        main_path = main_path
        name = "deeplodocus_project"

        if len(self.argv) > 2:
            name = self.argv[2]
        if len(self.argv) > 3:
            main_path = self.argv[3]

        p = ProjectUtility(project_name=name, main_path=main_path)
        p.generate_structure()
