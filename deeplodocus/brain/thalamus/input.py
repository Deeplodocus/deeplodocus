# import sys
# import asyncio
#
#
# def got_stdin_data(q):
#     asyncio.async(q.put(sys.stdin.readline()))
#
# class EchoServerClientProtocol(asyncio.Protocol):
#    def connection_made(self, transport):
#        peername = transport.get_extra_info('peername')
#        print('Connection from {}'.format(peername))
#        self.transport = transport
#
#    def data_received(self, data):
#        message = data.decode()
#        print('Data received: {!r}'.format(message))
#        fut = asyncio.async(q.get())
#        fut.add_done_callback(self.write_reply)
#
#    def write_reply(self, fut):
#        reply = fut.result()
#        print('Send: {!r}'.format(reply))
#        self.transport.write(reply.encode())
#
#        #print('Close the client socket')
#        #self.transport.close()
#
# q = asyncio.Queue()
# loop = asyncio.get_event_loop()
# loop.add_reader(sys.stdin, got_stdin_data, q)
# # Each client connection will create a new protocol instance
# coro = loop.create_server(EchoServerClientProtocol, '127.0.0.1', 8888)
# server = loop.run_until_complete(coro)
#
# # Serve requests until CTRL+c is pressed
# print('Serving on {}'.format(server.sockets[0].getsockname()))
# try:
#     loop.run_forever()
# except KeyboardInterrupt:
#     pass
#
# # Close the server
# server.close()
# loop.run_until_complete(server.wait_closed())
# loop.close()

import asyncio
import sys

x = 0

async def test(loop):
    global x
    while True:
        x = await loop.run_in_executor(None, add(x))

async def add(x):
    return x + 1

async def aio_readline(loop):
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        print('Got line:', line, end='')
        print(x)

loop = asyncio.get_event_loop()
loop.run_until_complete(test(loop))
loop.run_until_complete(aio_readline(loop))
loop.close()