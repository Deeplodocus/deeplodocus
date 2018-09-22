#
import os
import sys

#


#
from deeplodocus.utils.notification import Notification, DEEP_INFO, DEEP_ERROR

class ManagementUtility(object):


    def __init__(self, argv=None):

        self.commands = {"help" : "List the commands available", "version" : "Display the version of Deeplodocus installed", "startproject" : "Generate a deeplodocus to use Deeplodocus"}

        self.argv = argv or sys.argv[:]
        self.prog_name = os.path.basename(self.argv[0])
        if self.prog_name == '__main__.py':
            self.prog_name = 'python -m django'
        self.settings_exception = None


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
        Notification(DEEP_ERROR, "Not Implemented")


    def __startproject(self):
        """
        Authors : Alix Leroy,
        Generate the deeplodocus to use deeplodocus
        :return: None
        """

        self.__generate_config()

        self.__generate_main()

        self.__generate_loss()

        self.__generate_metric()

        self.__generate_result()




m = ManagementUtility(["help"])

m.execute_from_command_line()