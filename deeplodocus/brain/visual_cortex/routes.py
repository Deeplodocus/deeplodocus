import aiohttp
from aiohttp import web

from deeplodocus.brain.visual_cortex.views import index, test, monitor
from deeplodocus.utils.flags import *

class Routes(object):


    def __init__(self):
        self.list_routes = self.__load_routes()

    def setup_routes(self, app: aiohttp.web.Application, project_root):
        """
        Authors : Alix Leroy,
        Add the routes to the app
        :param app: The app to which we add the routes
        :return: None
        """
        for route in self.list_routes:
            app.router.add_route(route[0], route[1], route[2], name=route[3])

        # Add static content (css, js, etc...)
        app.router.add_static("/", path= str(project_root / "static"), name='static')

    def __load_routes(self):
        """
        Authors : Alix Leroy,
        Load in memory the list of routes
        :return: The list of routes
        """

        routes = [
            ('GET', '/', index, 'homepage'),            # Homepage
            ('GET', "/test", test, "test-page"),         # An example page
            ('GET', "/monitor", monitor, "monitor")
        ]

        return routes


