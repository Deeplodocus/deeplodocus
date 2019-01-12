from deeplodocus.utils.flag import Flag

# List the method used to load the data in the dataset

DEEP_LOAD_METHOD_MEMORY = 0
DEEP_LOAD_METHOD_HARDDRIVE = 1
DEEP_LOAD_METHOD_SERVER = 2

# List all the methods
DEEP_LOAD_METHOD_LIST = [DEEP_LOAD_METHOD_MEMORY,
                         DEEP_LOAD_METHOD_HARDDRIVE,
                         DEEP_LOAD_METHOD_SERVER]

# List the method used to load the data in the dataset

# High level methods
DEEP_LOAD_METHOD_ONLINE = Flag(name="Online",
                               description="Load the data during training",
                               names=["default", "online", "on line", "on-line", "on_line"])
DEEP_LOAD_METHOD_SEMI_ONLINE = Flag(name="Semi-online",
                                    description="Load the source before the training and the data during the training",
                                    names=["semi-online", "semionline", "semi online", "semi_online"])
DEEP_LOAD_METHOD_OFFLINE = Flag(name="Offline",
                                description="Load the data before training",
                                names=["offline", "off line", "off-line", "off_line", "memory"])

# Low level methods
DEEP_LOAD_METHOD_MEMORY_ = Flag(name="Memory", description="Load from memory", names=["memory"])
DEEP_LOAD_METHOD_HARD_DRIVE_ = Flag(name="Hard-drive", description="Load from hard drive", names=["default",
                                                                                                  "harddrive",
                                                                                                  "hard drive",
                                                                                                  "hard-drive",
                                                                                                  "hard_drive",
                                                                                                  "hdd",
                                                                                                  "hd"])
DEEP_LOAD_METHOD_SERVER_ = Flag(name="Server", description="Load from a remote server", names=["server"])
