# Python modules
import multiprocessing.managers
import weakref

# Deeplodocus modules
from deeplodocus.utils.flags.notif import *
from deeplodocus.utils.notification import Notification
from deeplodocus.brain.signal import Signal
from deeplodocus.utils.singleton import Singleton
from deeplodocus.brain.connection import Connection

class Thalamus(metaclass=Singleton):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Thalamus class.
    Manages and relays signals and events
    Inherits from Singleton : Only one unique instance of the class exists while running.
    Can be called anywhere in Deeplodocus

    TODO : Has to be tested when communicating with the visual cortex
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
        self.connections = {}                          # Connections store in a dictionary
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

        # To keep like this until asynchronous brain is implemented
        signal = self.signals.get()
        self.send(signal)

    def connect(self, receiver: callable, event: int, expected_arguments = None):
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

        connection = Connection(receiver=receiver, expected_arguments=expected_arguments)

        # If the event already register then add the reference to the set
        if event in self.connections:
            self.connections[event].add(connection)
        # Else initiate the list with the reference
        else:
            self.connections[event] = set([connection]) # Make a list around connection to make it iterable

    def disconnect(self, receiver: callable, event: int):
        """
        AUTHORS:
        --------

        :author: Alix Leroy


        DESCRIPTION:
        ------------

        Disconnect a receiver from a particular event

        PARAMETERS:
        -----------

        :param receiver(callable): The method to call on the event is fired
        :param event(int): The type of event

        RETURN:
        -------

        :return:
        """
        disconnected = False
        # For all the connections at the specific event
        for i, connection in enumerate(self.connections[event]):
            weak_ref_receiver = connection.get_receiver()

            # If the weak connection if found, we remove the connection
            if weak_ref_receiver in weakref.getweakrefs(receiver):
                self.connections[event][i].pop()
                disconnected = True
                break

        if disconnected is False:
            Notification(DEEP_NOTIF_ERROR, "The following receiver %s could not be disconnected from %i." %(str(receiver), event))


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
        # Get event id and arguments from the signal to send
        event = signal.get_event()
        args = signal.get_arguments()

        # If the event in the list broadcast the signal
        if event in self.connections:
            for connection in self.connections[event]:
                receiver = connection.get_receiver()
                expected_arguments = connection.get_expected_arguments()

                # If only some specific keys have to be kept
                if expected_arguments is not None:
                    args = self.keep_arguments(receiver=receiver, expected_arguments=expected_arguments, arguments=args)
                receiver()(**args)  # Need twice the brackets because of the weak method reference

        # Else display an error notification
        else:
            Notification(DEEP_NOTIF_ERROR, "The following event %s is not connected to any receiver." % str(event))

    def keep_arguments(self, receiver: callable, expected_arguments: list, arguments: dict):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Keep the desired arguments of a dictionary


        PARAMETERS:
        -----------

        :param receiver (callable): The function to receive the signal
        :param expected_arguments (list): The list of desired arguments
        :param arguments (dict): The dictionary to filter

        RETURN:
        -------

        :return kept_args(dict): The desired arguments
        """
        try:
            kept_args = {key: arguments[key] for key in expected_arguments}
            return kept_args
        except:
            print(receiver)
            print(arguments)
            print(expected_arguments)
