# Python imports
import sys
import os
import inspect
import shutil

# Deeplodocus imports
from deeplodocus import __version__
from deeplodocus.brain import Brain
from deeplodocus.core.project.project_utility import ProjectUtility
from deeplodocus.core.project.generators import *
from deeplodocus.core.project.structure.transformers import *
from deeplodocus.utils.generic_utils import get_module
from deeplodocus.utils.notification import Notification

# Deeplodocus flags
from deeplodocus.flags import *


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
            elif DEEP_ADMIN_IMPORT.corresponds((str(self.argv[1]))):
                self.__import(*self.argv[2:])
            else:
                Notification(
                    DEEP_NOTIF_ERROR,
                    "The following command does not exist : %s" % str(self.argv[1]),
                    log=False
                )
        else:
            self.__help()

    def __import(self, *args):
        for arg in args:
            items = []
            for m in DEEP_LIST_MODULE:
                method, module_path = get_module(arg, browse=m)
                if method is not None:
                    items.append(
                        {
                            "method": method,
                            "module_path": module_path,
                            "file": inspect.getsourcefile(method),
                            "prefix": m["custom"]["prefix"],
                            "name": arg
                        }
                    )
            if not items:
                Notification(DEEP_NOTIF_ERROR, "module not found : %s" % arg)
            elif len(items) == 1:
                self.__import_module(items[0])
            else:
                Notification(DEEP_NOTIF_INFO, "Multiple modules found with the same name :")
                for i, item in enumerate(items):
                    Notification(DEEP_NOTIF_INFO, "  %i. %s from %s" % (i, arg, item["module_path"]))
                while True:
                    i = Notification(DEEP_NOTIF_INPUT, "Which would you like?").get()
                    try:
                        i = int(i)
                        break
                    except ValueError:
                        pass
                self.__import_module(items[i])

    @staticmethod
    def __import_module(m):
        directory = m["prefix"].replace(".", "/")
        path = "/".join((directory, m["file"].split("/")[-1]))
        os.makedirs(directory, exist_ok=True)
        shutil.copy(m["file"], path)
        Notification(DEEP_NOTIF_SUCCESS, "Imported %s to %s" % (m["name"], path))

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
        directory = self.setup_transformers_directory()
        filename = self.__generate_filename(filename)
        generate_transformer(OUTPUT_TRANSFORMER, filename="/".join((directory, filename)))
        Notification(DEEP_NOTIF_SUCCESS, "New output transformer file created : %s" % filename, log=False)

    def __oneof_transformer(self):
        try:
            filename = self.argv[2]
        except IndexError:
            filename = "oneof_transformer.yaml"
        directory = self.setup_transformers_directory()
        filename = self.__generate_filename(filename)
        generate_transformer(ONEOF_TRANSFORMER, filename="/".join((directory, filename)))
        Notification(DEEP_NOTIF_SUCCESS, "New one-of transformer file created : %s" % filename, log=False)

    def __sequential_transformer(self):
        try:
            filename = self.argv[2]
        except IndexError:
            filename = "sequential_transformer.yaml"
        directory = self.setup_transformers_directory()
        filename = self.__generate_filename(filename)
        generate_transformer(SEQUENTIAL_TRANSFORMER, filename="/".join((directory, filename)))
        Notification(DEEP_NOTIF_SUCCESS, "New sequential transformer file created : %s" % filename, log=False)

    def __someof_transformer(self):
        try:
            filename = self.argv[2]
        except IndexError:
            filename = "someof_transformer.yaml"
        directory = self.setup_transformers_directory()
        filename = self.__generate_filename(filename)
        generate_transformer(SOMEOF_TRANSFORMER, filename="/".join((directory, filename)))
        Notification(DEEP_NOTIF_SUCCESS, "New some-of transformer file created : %s" % filename, log=False)

    def __run_project(self):
        Notification(DEEP_NOTIF_WARNING, "The run project command does not support custom modules")
        Notification(DEEP_NOTIF_WARNING, "If you need to import custom modules use : python3 main.py")
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

    @staticmethod
    def setup_transformers_directory(directory="config/transformers"):
        os.makedirs(directory, exist_ok=True)
        return directory
