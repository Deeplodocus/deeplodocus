from deeplodocus.utils.flags.ext import DEEP_EXT_YAML

# The divider to use when expressing paths to configs
DEEP_CONFIG_DIVIDER = "/"

# Keywords
DEEP_CONFIG_ENABLED = "enabled"
DEEP_CONFIG_DEFAULT = "default"
DEEP_CONFIG_DTYPE = "dtype"
DEEP_CONFIG_WILDCARD = "*"

# Names of each section of the config fle
DEEP_CONFIG_PROJECT = "project"
DEEP_CONFIG_MODEL = "model"
DEEP_CONFIG_TRAINING = "training"
DEEP_CONFIG_DATA = "data"
DEEP_CONFIG_TRANSFORM = "transform"
DEEP_CONFIG_OPTIMIZER = "optimizer"
DEEP_CONFIG_METRICS = "metrics"
DEEP_CONFIG_LOSS = "losses"
DEEP_CONFIG_HISTORY = "history"

# List of all config sections
DEEP_CONFIG_SECTIONS = [DEEP_CONFIG_PROJECT,
                        DEEP_CONFIG_MODEL,
                        DEEP_CONFIG_TRAINING,
                        DEEP_CONFIG_DATA,
                        DEEP_CONFIG_TRANSFORM,
                        DEEP_CONFIG_OPTIMIZER,
                        DEEP_CONFIG_METRICS,
                        DEEP_CONFIG_LOSS,
                        DEEP_CONFIG_HISTORY]

# Define the expected structure of the project configuration space
# First level keys define the name of config .yaml files
# Second level keys and below should be found in each file
# Default values and  data types must be given for each configuration
# NB: if a list of floats is expected, use [float] instead of float
DEEP_CONFIG = {DEEP_CONFIG_PROJECT: {"name": {"dtype": str,
                                              "default": "deeplodocus_project"},
                                     "cv_library": {"dtype": str,
                                                    "default": "pil"},
                                     "logs": {"history_train_batches": {"dtype": bool,
                                                                        "default": True},
                                              "history_train_epochs": {"dtype": bool,
                                                                       "default": True},
                                              "history_validation": {"dtype": bool,
                                                                     "default": True},
                                              "notification": {"dtype": bool,
                                                               "default": True}
                                              },
                                     "on_wake": {"dtype": [str],
                                                 "default": None}
                                     },
               DEEP_CONFIG_MODEL: {"module": {"dtype": str,
                                   "default": None},
                                    "name": {"dtype": str,
                                            "default": "LeNet"},
                                    "input_size": {"dtype" : [[int]],
                                                   "default" : [[1, 1]]},
                                   "kwargs": {"dtype": dict,
                                              "default": None}
                                   },
               DEEP_CONFIG_OPTIMIZER: {"module": {"dtype": str,
                                       "default": None},
                                       "name": {"dtype": str,
                                                "default": "Adam"},
                                       "kwargs": {"dtype": dict,
                                                  "default": {"lr": 0.001,
                                                              "eps": 0.000000001,
                                                              "amsgrad": False,
                                                              "betas": [0.9, 0.999],
                                                              "weight_decay": 0.0}
                                                  }
                                       },
               DEEP_CONFIG_HISTORY: {"verbose": {"dtype": int,
                                                 "default": 1},
                                     "memorize": {"dtype": int,
                                                  "default": 1}},
               DEEP_CONFIG_TRAINING: {"num_epochs": {"dtype": int,
                                                     "default": 10},
                                      "initial_epoch": {"dtype": int,
                                                        "default": 0},
                                      "shuffle": {"dtype": int,
                                                  "default": 2},
                                      "save_condition": {"dtype": int,
                                                         "default": 1},
                                      "save_method": {"dtype": int,
                                                      "default": 1}
                                      },
               DEEP_CONFIG_DATA: {"dataloader": {"batch_size": {"dtype": int,
                                                                "default": 32},
                                                 "num_workers": {"dtype": int,
                                                                 "default": 1}
                                                 },
                                  DEEP_CONFIG_ENABLED: {"train": {"dtype": bool,
                                                                  "default": True},
                                                        "validation": {"dtype": bool,
                                                                       "default": True},
                                                        "test": {"dtype": bool,
                                                                 "default": True}
                                                        },
                                  "dataset": {"train": {"inputs": {"dtype": [{"source": [str],
                                                                              "join": [str],
                                                                              "type": str,
                                                                              "load_method": str}],
                                                                   "default": None},
                                                        "labels": {"dtype": [{"source": [str],
                                                                              "join": str,
                                                                              "type": str,
                                                                              "load_method": str}],
                                                                   "default": None},
                                                        "additional_data": {"dtype": [{"source": [str],
                                                                                       "join": str,
                                                                                       "type": str,
                                                                                       "load_method": str}],
                                                                            "default": None},
                                                        "number": {"dtype": int,
                                                                   "default": None},
                                                        "name": {"dtype": str,
                                                                 "default": "train"}
                                                        },
                                              "validation": {"inputs": {"dtype": [{"source": [str],
                                                                                   "join": str,
                                                                                   "type": str,
                                                                                   "load_method": str}],
                                                                        "default": None},
                                                             "labels": {"dtype": [{"source": [str],
                                                                                   "join": str,
                                                                                   "type": str,
                                                                                   "load_method": str}],
                                                                        "default": None},
                                                             "additional_data": {"dtype": [{"source": [str],
                                                                                            "join": str,
                                                                                            "type": str,
                                                                                            "load_method": str}],
                                                                                 "default": None},
                                                             "number": {"dtype": int,
                                                                        "default": None},
                                                             "name": {"dtype": str,
                                                                      "default": "validation"}
                                                             },
                                              "test": {"inputs": {"dtype": [{"source": [str],
                                                                             "join": str,
                                                                             "type": str,
                                                                             "load_method": str}],
                                                                  "default": None},
                                                       "labels": {"dtype": [{"source": [str],
                                                                             "join": str,
                                                                             "type": str,
                                                                             "load_method": str}],
                                                                  "default": None},
                                                       "additional_data": {"dtype": [{"source": [str],
                                                                                      "join": str,
                                                                                      "type": str,
                                                                                      "load_method": str}],
                                                                           "default": None},
                                                       "number": {"dtype": int,
                                                                  "default": None},
                                                       "name": {"dtype": str,
                                                                "default": "test"}
                                                       }
                                              }
                                  },
               DEEP_CONFIG_LOSS: {DEEP_CONFIG_WILDCARD: {"module": {"dtype": str,
                                                         "default": None},
                                                         "name": {"dtype": str,
                                                                  "default": "CrossEntropyLoss"},
                                                         "weight": {"dtype": float,
                                                                    "default": 1},
                                                         "kwargs": {"dtype": dict,
                                                                    "default": {}}
                                                         }
                                  },
               DEEP_CONFIG_TRANSFORM: {},
               DEEP_CONFIG_METRICS: {DEEP_CONFIG_WILDCARD: {"module": {"dtype": str,
                                                                       "default": None},
                                                            "name": {"dtype": str,
                                                                     "default": "CrossEntropyLoss"},
                                                            "weight": {"dtype": float,
                                                                       "default": 1},
                                                            "kwargs": {"dtype": dict,
                                                                       "default": {}}
                                                            }
                                     }
               }

# A dict of names for each config file
DEEP_CONFIG_FILES = {item: "%s%s" % (item, DEEP_EXT_YAML) for item in DEEP_CONFIG_SECTIONS}
