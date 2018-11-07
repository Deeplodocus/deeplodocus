from multiprocessing import Process
import sys

from aiohttp import web
import aiohttp_jinja2
import jinja2

from deeplodocus.utils.notification import Notification
from deeplodocus.ui.routes import Routes
from deeplodocus.utils.flags import *


class UserInterface(object):

    def __init__(self):

        # Create the web server in a second process
        self.process = Process(target=self.__run, args=())

        self.process.daemon = True # Allow to kill the child with the parent
        self.process.start()
        #self.process.join()

    def __run(self):
        """
        Authors : Alix Leroy,
        Run the web server
        :return: None
        """

        Notification(DEEP_NOTIF_SUCCESS, "User Interface created")

        app = web.Application()                                                                     # Start the web application
        aiohttp_jinja2.setup(app, loader = jinja2.PackageLoader('deeplodocus', 'ui/templates'))     # Load the templates
        Routes().setup_routes(app=app)                                                              # Define the routes
        web.run_app(app)                                                                            # Run the app

        Notification(DEEP_NOTIF_SUCCESS, "User interface closed successfully")
        sys.exit(0)  # kill the child process


    def stop(self):
        self.process.terminate() # Terminate the process




