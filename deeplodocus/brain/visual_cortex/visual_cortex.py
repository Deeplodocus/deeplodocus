# Python Modules
from multiprocessing import Process
import sys
import pathlib
import time

# Web server Modules
from aiohttp import web
import aiohttp_jinja2
import jinja2

# Deeplodocus modules
from deeplodocus.utils.notification import Notification
from deeplodocus.brain.visual_cortex.routes import Routes
from deeplodocus.brain.visual_cortex.middlewares import setup_middlewares

# Deeplodocus flags

class VisualCortex(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    A User Interface accessible via a Web Browser

    TODO : Include Plotly for better visualization

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
        self.process.daemon = True  # Allow to kill the child with the parent
        self.process.start()
        # self.process.join()
        time.sleep(0.5)


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

        VISUAL_CORTEX_ROOT = pathlib.Path(__file__).parent
        host = "0.0.0.0"
        port = 8080
        Notification(DEEP_NOTIF_SUCCESS, "Brain : Visual Cortex running on : http://%s:%i" %(host, port))
        app = web.Application()                                                                     # Start the web application
        aiohttp_jinja2.setup(app, loader = jinja2.PackageLoader('deeplodocus', 'brain/visual_cortex/templates'))     # Load the templates
        Routes().setup_routes(app=app, project_root=VISUAL_CORTEX_ROOT)                            # Define the routes
        setup_middlewares(app)                                                                      # Define the middlewares

        web.run_app(app=app, print=None, host=host, port=port)                                      # Run the app

        Notification(DEEP_NOTIF_SUCCESS, "Visual Cortex sleeping.")
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





