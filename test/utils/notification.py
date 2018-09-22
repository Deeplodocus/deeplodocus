"""
Authors : Alix Leroy
Test the different notification available
"""

from deeplodocus.utils.notification import Notification, DEEP_INFO, DEEP_SUCCESS, DEEP_ERROR, DEEP_INPUT, DEEP_FATAL, DEEP_WARNING, DEEP_DEBUG

Notification(DEEP_WARNING, "Warning notification")
Notification(DEEP_INFO, "Info notification")
Notification(DEEP_SUCCESS, "Success notification")
Notification(DEEP_ERROR, "Error notification")
Notification(DEEP_DEBUG, "Debug notification")
Notification(DEEP_INPUT, "Input notification, please enter anything to proceed")
Notification("Not a good type", "Default notification")
Notification(DEEP_FATAL, "Fatal error exiting the program")