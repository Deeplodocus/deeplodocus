# Python imports
import sys
import os
import datetime
import time

# Deeplodocus imports
from deeplodocus.utils.notification import Notification
from deeplodocus.core.project.project_utility import ProjectUtility
from deeplodocus import __version__

# Deeplodocus flags
from deeplodocus.utils.flags.notif import *
from deeplodocus.utils.flags.admin import *
from deeplodocus.utils.flags.flag_lists import DEEP_LIST_ADMIN


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

            if DEEP_ADMIN_START_PROJECT.corresponds(str(self.argv[1])) :
                self.__startproject(main_path=os.getcwd())

            elif DEEP_ADMIN_VERSION.corresponds(str(self.argv[1])):
                self.__version()

            elif DEEP_ADMIN_HELP.corresponds(str(self.argv[1])):

                self.__help()

            else:
                Notification(DEEP_NOTIF_ERROR, "The following command does not exits : " + str(self.argv[1]),  log=False)

        else:
            self.__help()

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

    def __startproject(self, main_path: str):
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
        name = "deeplodocus_project_" + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')

        if len(self.argv) > 2:
            name = self.argv[2]
        if len(self.argv) > 3:
            main_path = self.argv[3]

        p = ProjectUtility(project_name=name, main_path=main_path)
        p.generate_structure()
