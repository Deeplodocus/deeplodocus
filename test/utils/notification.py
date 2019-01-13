"""
Authors : Alix Leroy
Test the different notification available
"""

from deeplodocus.utils.flags.notif import *
from deeplodocus.utils.notification import Notification

Notification(DEEP_NOTIF_WARNING, "Warning notification", log=False)
Notification(DEEP_NOTIF_INFO, "Info notification", log=False)
Notification(DEEP_NOTIF_SUCCESS, "Success notification", log=False)
Notification(DEEP_NOTIF_ERROR, "Error notification", log=False)
Notification(DEEP_NOTIF_DEBUG, "Debug notification", log=False)
Notification(DEEP_NOTIF_INPUT, "Input notification, please enter anything to proceed", log=False)
Notification(DEEP_NOTIF_LOVE, "Love notif when exiting the program", log=False)
Notification(DEEP_NOTIF_FATAL, "Fatal error returning to the main loop", log=False)
