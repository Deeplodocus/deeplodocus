import weakref


class Connection(object):


    def __init__(self, receiver: callable, expected_arguments=None):
        # Get the weak reference of the method to call

        self.receiver = weakref.WeakMethod(receiver)
        self.expected_arguments = expected_arguments

    def get_receiver(self):
        return self.receiver

    def get_expected_arguments(self):
        return self.expected_arguments