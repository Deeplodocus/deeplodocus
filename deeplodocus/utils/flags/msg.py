# DEEP MESSAGES / TEXT / STATEMENTS


# Deep Error
DEEP_MSG_FILE_NOT_FOUND = "File not found : %s"
DEEP_MSG_DIR_NOT_FOUND = "Directory not found : %s"
DEEP_MSG_ILLEGAL_COMMAND = "Illegal command : %s"
DEEP_MSG_NO_TESTER = "Cannot evaluate : Tester not loaded"
DEEP_MSG_NO_VALIDATOR = "Cannot validate : Validator not loaded"
DEEP_MSG_NO_PREDICTOR = "Cannot validate : Predictor not loaded"
DEEP_MSG_NO_TRAINER = "Cannot evaluate : Trainer not loaded"
DEEP_MSG_INVALID_DEVICE = "%s is not a valid input device : Please specify 'cuda' or 'cpu'"

# Deep Success
DEEP_MSG_LOAD_CONFIG_FILE = "File loaded : %s"
DEEP_MSG_PROJECT_GENERATED = "Project successfully generated ! Have fun <3"
DEEP_MSG_MODULE_LOADED = "Module loaded : %s from %s"

# Deep Warning
DEEP_MSG_ALREADY_AWAKE = ": I am already awake !"
DEEP_MSG_USE_CONFIG_SAVE = ": Please use 'save_config()' instead"
DEEP_MSG_PRIVATE = ": Please don't interfere with my private parts"
DEEP_MSG_PROJECT_ALREADY_EXISTS = ": Project %s already exists !"

# Deep Input
DEEP_MSG_INSTRUCTRION = "Awaiting instruction ..."
DEEP_MSG_CONTINUE = "Do you wish to continue ? (y/n)"

# Deep Info
DEEP_MSG_PROJECT_NOT_GENERATED = "Project not generated"


#########
# SAVER #
#########

DEEP_MSG_SAVER_IMPROVED = "%s improved by %s : Saving weights"
DEEP_MSG_SAVER_NOT_IMPROVED = "No improvement in %s : Not saving weights"

#########
# BRAIN #
#########

DEEP_MSG_BRAIN_CLEAR_ALL = "Clearing files from %s"
DEEP_MSG_BRAIN_CLEARED_ALL = "Files cleared from %s"
DEEP_MSG_BRAIN_CLEAR_HISTORY = "Clearing files from %s/history"
DEEP_MSG_BRAIN_CLEARED_HISTORY = "Files cleared from %s/history"
DEEP_MSG_BRAIN_CLEAR_LOGS = "Clearing files from %s/logs"
DEEP_MSG_BRAIN_CLEARED_LOGS = "Files cleared from %s/logs"

####################
# DEEP_MSG_PROJECT #
####################

# DEEP_SUCCESS
DEEP_MSG_PROJECT_DEVICE = "Device set : %s"

# DEEP_FATAL
DEEP_MSG_PROJECT_DEVICE_NOT_FOUND = "Device not found : %s"

###################
# DEEP_MSG_CONFIG #
###################

# DEEP_INFO
DEEP_MSG_CONFIG_LOADING_DIR = "Loading configurations : %s"
DEEP_MSG_CONFIG_LOADING_FILE = "Loading configuration file : %s"

# DEEP_SUCCESS
DEEP_MSG_CONFIG_COMPLETE = "All configurations complete"

# DEEP_WARNING
DEEP_MSG_CONFIG_NOT_FOUND = "Entry not found : %s : Using default value : %s"
DEEP_MSG_CONFIG_NOT_SET = "Entry not set : %s : Using default value : %s"
DEEP_MSG_CONFIG_NOT_CONVERTED = "Type error : At %s : Could not convert %s to type %s : Using default value : %s"

#################
# DEEP_MSG_DATA #
#################

# DEEP_FATAL
DEEP_MSG_DATA_CANNOT_IDENTIFY_IMAGE = "%s could not identify image file : %s"
DEEP_MSG_DATA_CANNOT_FIND_IMAGE = "Image not found : %s"
DEEP_MSG_DATA_CANNOT_LOAD_IMAGE = "%s could not open image file %s"
DEEP_MSG_DATA_NOT_HANDLED = "The type of the following data is not handled : %s"
DEEP_MSG_DATA_SOURCE_NOT_FOUND = "Source path not found : %s"
DEEP_MSG_DATA_IS_NONE = "The following data is None : %s"
DEEP_MSG_DATA_ENTRY = "Please check the following entry format : %s"
DEEP_MSG_DATA_INDEX_ERROR = "No input entries given for %s dataset"

# DEEP WARNING
DEEP_MSG_DATA_SHORTER = "Dataset contains %i instances : Using just %i instances"
DEEP_MSG_DATA_TOO_LONG = "Dataset number %i exceeds the number of instances : Using just %i instances"

# DEEP_INFO
DEEP_MSG_DATA_SUMMARY = "Summary of the '%s' dataset :\n%s"
DEEP_MSG_DATA_LENGTH = "Dataset length set to %i instances"
DEEP_MSG_DATA_NO_LENGTH = "Dataset length not given : Using all %i instances"
DEEP_MSG_DATA_DISABLED = "Dataset disabled : %s"
DEEP_NOTIF_DATA_LOADING = "Loading dataset : %s"
DEEP_MSG_DATA_GREATER = "Dataset number (%i) is greater than the number of instances (%i): Additional data will be transformed"
DEEP_MSG_DATA_INDEX_ERROR_SOLUTION_1 = "Make sure inputs are given for your %s dataset in config/data/dataset"
DEEP_MSG_DATA_INDEX_ERROR_SOLUTION_2 = "Disable your %s dataset in config/data/enabled"

# DEEP_SUCCESS
DEEP_MSG_DATA_LOADED = "Dataset loaded : %s"

######################
# DEEP_MSG_TRANSFORM #
######################

# DEEP_FATAL
DEEP_MSG_TRANSFORM_VALUE_ERROR = "Transforms must return two items"

#####################
# DEEP_MSG_TRAINING #
#####################

DEEP_MSG_TRAINING_STARTED = "Started training"
DEEP_MSG_TRAINING_FINISHED = "Finished training !"

#######################
# DEEP_MSG_CV_LIBRARY #
#######################
DEEP_MSG_CV_LIBRARY_SET = "Set cv library to : %s"
DEEP_MSG_CV_LIBRARY_NOT_IMPLEMENTED = "The following image module is not implemented : %s"

##################
# DEEP_MSG_NOTIF #
##################

# DEEP_FATAL
DEEP_MSG_NOTIF_UNKNOWN = "Unknown notification type : %s"

##################
# DEEP_MSG_MODEL #
##################

# DEEP_FATAL
DEEP_MSG_MODEL_NOT_FOUND = "Model not found : %s"
DEEP_MSG_MODEL_NO_FILE = "No model file was given"
DEEP_MSG_MODEL_FILE_NOT_FOUND = "Model file not found : %s"
DEEP_MSG_MODEL_CHECK_CHANNELS = "Check that the config/data/input_size corresponds to the number of channels given in config/data/kwargs"

# DEEP_SUCCESS
DEEP_MSG_MODEL_LOADED = "Model loaded : %s"

# DEEP_INFO
DEEP_MSG_MODEL_LOADING = "Loading model : %s"
DEEP_MSG_MODEL_SAVED = "Model and weights saved to : %s"

##################
# DEEP_MSG_OPTIM #
##################

# DEEP_SUCCESS
DEEP_MSG_OPTIM_LOADED = "Optimizer loaded : %s"

# DEEP_INFO
DEEP_MSG_OPTIM_LOADING = "Loading optimizer : %s"
DEEP_MSG_OPTIM_NOT_LOADED = "Optimizer not loaded"

# DEEP_FATAL
DEEP_MSG_OPTIM_NOT_FOUND = "Optimizer not found : %s"
DEEP_MSG_OPTIM_MODEL_NOT_LOADED = "Could not load optimizer : Model not loaded"


#################
# DEEP_MSG_LOSS #
#################

# DEEP_FATAL
DEEP_MSG_LOSS_NOT_FOUND = "Loss not found : %s"
DEEP_MSG_LOSS_NOT_TORCH = "Loss is not a torch.nn.Module instance : %s : %s from %s"

# DEEP_SUCCESS
DEEP_MSG_LOSS_LOADED = "Loss loaded : %s : %s from %s"

# DEEP_INFO
DEEP_MSG_LOSS_LOADING = "Loading loss : %s"
DEEP_MSG_LOSS_NONE = "No losses to load"
DEEP_MSG_LOSS_NOT_LOADED = "Losses not loaded"

###################
# DEEP_MSG_METRIC #
###################

# DEEP_FATAL
DEEP_MSG_METRIC_NOT_FOUND = "Metric not found : %s"

# DEEP_SUCCESS
DEEP_MSG_METRIC_LOADED = "Metric loaded : %s : %s from %s"

# DEEP_INFO
DEEP_MSG_METRIC_LOADING = "Loading metrics : %s"
DEEP_MSG_METRIC_NONE = "No metrics to load"
DEEP_MSG_METRIC_NOT_LOADED = "Metrics not loaded"

####################
# DEEP_MSG_SHUFFLE #
####################

DEEP_MSG_SHUFFLE_NOT_FOUND = "Shuffling method does not exist : %s"
DEEP_MSG_SHUFFLE_COMPLETE = "Dataset shuffled with method : %s"

