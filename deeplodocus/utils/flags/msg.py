from deeplodocus.utils.flags.config import DEEP_CONFIG_DIVIDER, DEEP_CONFIG_PROJECT

#
# DEEP MESSAGES / TEXT / STATEMENTS
#

# Error messages
DEEP_MSG_FILE_NOT_FOUND = "File not found : %s"
DEEP_MSG_DIR_NOT_FOUND = "Directory not found : %s"
DEEP_MSG_ILLEGAL_COMMAND = "Illegal command : %s"
DEEP_MSG_CONFIG_NOT_FOUND = "Configuration not found : %s"
DEEP_MSG_LOAD_CONFIG_FAIL = "Missing configurations : Please include the configurations listed above"
DEEP_MSG_MODEL_NOT_FOUND = "Model %s not found in %s."
DEEP_MSG_LOSS_NOT_FOUND = "Loss not found : %s"
DEEP_MSG_METRIC_NOT_FOUND = "Metric not found : %s"



# Success messages
DEEP_MSG_LOAD_CONFIG_FILE = "File loaded : %s"
DEEP_MSG_LOAD_CONFIG_SUCCESS = "All necessary configurations have been imported"

# Info messages
DEEP_MSG_ALREADY_AWAKE = ": I am already awake !"
DEEP_MSG_INSTRUCTRION = "Awaiting instruction ..."
DEEP_MSG_LOAD_CONFIG_START = "Loading configurations from %s"
DEEP_MSG_REMOVE_LOGS = "%s%swrite_logs is False : Notification logs have been removed" % (DEEP_CONFIG_PROJECT,
                                                                                          DEEP_CONFIG_DIVIDER)
DEEP_MSG_USE_CONFIG_SAVE = ": Please use 'save_config(save_as=file_name)' instead"
