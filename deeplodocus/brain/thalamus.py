# Python modules
import weakref
import multiprocessing.managers

# Deeplodocus modules
from deeplodocus.utils.flags.notif import *
from deeplodocus.utils.notification import Notification
from deeplodocus.brain.signal import Signal

class Thalamus(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Thalamus class.
    Manages and relays signals
    """


    def __init__(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize the Thalamus.
        Create a queue of signals
        Initialize connections with a dictionary (values are sets)
        """
        self.signals = multiprocessing.Manager().Queue()                   # To be used once asynchronous brain is implemented
        self.connections = {}   # Connections store in a dictionary
        Notification(DEEP_NOTIF_SUCCESS, "Brain : Thalamus running")


    def add_signal(self, signal: Signal):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Add a signal to be processed

        PARAMETERS:
        -----------

        :param signal(SensorySignal): The signal to add

        RETURN:
        -------

        :return: None
        """
        self.signals.put(signal)

    def connect(self, receiver: callable, event: int):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Establish a connection between an organ and a specific event.

        PARAMETERS:
        -----------

        :param receiver(callable): The method to call when firing the signal
        :param event(int): The type of event

        RETURN:
        -------

        :return: None
        """
        # Get the weak reference of the method to call
        ref = weakref.ref(receiver)

        # If the event already register then add the reference to the set
        if event in self.connections:
            self.connections[event].add(ref)
        # Else initiate the list with the reference
        else:
            self.connections[event] = {ref}

    def disconnect(self, receiver: callable, event: int):
        pass
        # TODO : Disconnect a weak ref

    def send(self, signal: Signal):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Send the signal to the appropriate receivers

        PARAMETERS:
        -----------

        :param signal(Signal): A signal to send

        RETURN:
        -------

        :return: None
        """
        event = signal.get_event()
        args = signal.get_arguments()

        # If the event in the list broadcast the signal
        if event in self.connections:
            for receiver in self.connections[event]:
                receiver(**args)
        # Else display an error notification
        else:
            Notification(DEEP_NOTIF_ERROR, "The following event is not connected to any receiver.")
