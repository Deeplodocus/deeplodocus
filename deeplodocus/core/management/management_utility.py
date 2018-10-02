#
import os
import sys

#


#
from deeplodocus.utils.notification import Notification, DEEP_INFO, DEEP_ERROR
from deeplodocus.core.project.project_utility import ProjectUtility
from deeplodocus import __version__

class ManagementUtility(object):


    def __init__(self, argv=None):

        self.commands = {"help" : "List the commands available",
                         "version" : "Display the version of Deeplodocus installed",
                         "startproject" : "Generate a deeplodocus to use Deeplodocus"}

        self.argv = argv or sys.argv[:]


    def execute_from_command_line(self):
        """
        Authors : Alix Leroy,
        execute the command entered in the terminal
        :return: None
        """

        if len(self.argv) > 1:

            if str(self.argv[1]) == "startproject":

                self.__startproject()


            elif str(self.argv[1]) == "version":
                self.__version()


            elif str(self.argv[1]) == "help":

                self.__help()

            else:
                Notification(DEEP_ERROR, "The following command does not exits : " + str(self.argv[1]),  write_logs=False)

        else:
            self.__help()


    def __help(self):


        for command, description in self.commands.items():
            Notification(DEEP_INFO, str(command) + " : " + str(description), write_logs=False)

    def __version(self):

        version = str(__version__)
        Notification(DEEP_INFO, "DEEPLODOCUS VERSION : " + str(version),  write_logs=False)


    def __startproject(self):
        """
        Authors : Alix Leroy,
        Generate the deeplodocus structure to use deeplodocus
        :return: None
        """

        main_path = None
        name = "deeplodocus_project"

        if len(self.argv)>2:
            name = self.argv[2]
        if len(self.argv)>3:
            main__path = self.argv[3]

        p = ProjectUtility(project_name=name, main_path =main_path)
        p.generate_structure()
