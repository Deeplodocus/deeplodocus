# Python Modules
from multiprocessing import Process
import sys
import pathlib


# Web server Modules
from aiohttp import web
import aiohttp_jinja2
import jinja2

# Deeplodocus modules
from deeplodocus.utils.notification import Notification
from deeplodocus.ui.routes import Routes
from deeplodocus.utils.flags import *
from deeplodocus.ui.middlewares import setup_middlewares


class UserInterface(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    A User Interface accessible via a Web Browser
    """

    def __init__(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Start the user interface in a second Process linked to the main one.
        """

        # Create the web server in a second process
        self.process = Process(target=self.__run, args=())

        self.process.daemon = True # Allow to kill the child with the parent
        self.process.start()
        #self.process.join()

    def __run(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Run the web server

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        USER_INTERFACE_ROOT = pathlib.Path(__file__).parent

        Notification(DEEP_NOTIF_SUCCESS, "User Interface created")

        app = web.Application()                                                                     # Start the web application
        aiohttp_jinja2.setup(app, loader = jinja2.PackageLoader('deeplodocus', 'ui/templates'))     # Load the templates
        Routes().setup_routes(app=app, project_root=USER_INTERFACE_ROOT)                            # Define the routes
        setup_middlewares(app)
        web.run_app(app)                                                                            # Run the app

        Notification(DEEP_NOTIF_SUCCESS, "User interface closed successfully")
        sys.exit(0)  # kill the child process


    def stop(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Stop the server

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        self.process.terminate() # Terminate the process




