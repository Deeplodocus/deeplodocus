"""
Authors : Alix Leroy
Test the different notification available
"""

from deeplodocus.utils.notification import Notification

Notification(DEEP_NOTIF_WARNING, "Warning notification", write_logs=False)
Notification(DEEP_NOTIF_INFO, "Info notification", write_logs=False)
Notification(DEEP_NOTIF_SUCCESS, "Success notification", write_logs=False)
Notification(DEEP_NOTIF_ERROR, "Error notification", write_logs=False)
Notification(DEEP_NOTIF_DEBUG, "Debug notification", write_logs=False)
Notification(DEEP_NOTIF_INPUT, "Input notification, please enter anything to proceed", write_logs=False)
Notification("Not a good type", "Default notification", write_logs=False)
Notification(DEEP_NOTIF_FATAL, "Fatal error exiting the program", write_logs=False)