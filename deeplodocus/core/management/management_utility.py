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

        if self.argv[0] == "startproject":

            self.__startproject()


        elif self.argv[0] == "version":
            self.__version()


        elif self.argv[0] == "help":

            self.__help()

        else:
            Notification(DEEP_ERROR, "The following command does not exits : " + str(self.argv[0]))




    def __help(self):


        for command, description in self.commands.items():
            Notification(DEEP_INFO, str(command) + " : " + str(description))

    def __version(self):

        version = str(__version__)
        Notification(DEEP_INFO, "DEEPLODOCUS VERSION : " + str(version))


    def __startproject(self):
        """
        Authors : Alix Leroy,
        Generate the deeplodocus structure to use deeplodocus
        :return: None
        """

        p = ProjectUtility()
        p.generate_structure()




m = ManagementUtility(["help"])

m.execute_from_command_line()