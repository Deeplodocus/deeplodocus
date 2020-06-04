from aiohttp import web

import aiohttp_jinja2

@aiohttp_jinja2.template('index.html')
async def index(request):
    return {'name': 'Andrew',
            'surname': 'Svetlov'}

@aiohttp_jinja2.template('test.html')
async def test(request):
    return {'name': 'Andrew',
            'surname': 'Svetlov'}

@aiohttp_jinja2.template('monitor.html')
async def monitor(request):
    return {}