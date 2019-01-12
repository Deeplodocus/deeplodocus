from deeplodocus.utils.flags.ext import DEEP_EXT_YAML

# The divider to use when expressing paths to configs
DEEP_CONFIG_DIVIDER = "/"

# Keywords
DEEP_CONFIG_DEFAULT = "DEFAULT"
DEEP_CONFIG_DTYPE = "DTYPE"
DEEP_CONFIG_AUTO = "auto"
DEEP_CONFIG_ENABLED = "enabled"
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
DEEP_CONFIG = {DEEP_CONFIG_PROJECT: {"name": {DEEP_CONFIG_DTYPE: str,
                                              DEEP_CONFIG_DEFAULT: "deeplodocus_project"},
                                     "cv_library": {DEEP_CONFIG_DTYPE: str,
                                                    DEEP_CONFIG_DEFAULT: "pil"},
                                     "device": {DEEP_CONFIG_DTYPE: str,
                                                DEEP_CONFIG_DEFAULT: DEEP_CONFIG_AUTO},
                                     "gpus": {DEEP_CONFIG_DTYPE: str,
                                              DEEP_CONFIG_DEFAULT: DEEP_CONFIG_AUTO},
                                     "logs": {"history_train_batches": {DEEP_CONFIG_DTYPE: bool,
                                                                        DEEP_CONFIG_DEFAULT: True},
                                              "history_train_epochs": {DEEP_CONFIG_DTYPE: bool,
                                                                       DEEP_CONFIG_DEFAULT: True},
                                              "history_validation": {DEEP_CONFIG_DTYPE: bool,
                                                                     DEEP_CONFIG_DEFAULT: True},
                                              "notification": {DEEP_CONFIG_DTYPE: bool,
                                                               DEEP_CONFIG_DEFAULT: True}
                                              },
                                     "on_wake": {DEEP_CONFIG_DTYPE: [str],
                                                 DEEP_CONFIG_DEFAULT: None}
                                     },
               DEEP_CONFIG_MODEL: {"module": {DEEP_CONFIG_DTYPE: str,
                                              DEEP_CONFIG_DEFAULT: None},
                                   "name": {DEEP_CONFIG_DTYPE: str,
                                            DEEP_CONFIG_DEFAULT: "LeNet"},
                                   "input_size": {DEEP_CONFIG_DTYPE: [[int]],
                                                  DEEP_CONFIG_DEFAULT: [[None]]},
                                   "kwargs": {DEEP_CONFIG_DTYPE: dict,
                                              DEEP_CONFIG_DEFAULT: None}
                                   },
               DEEP_CONFIG_OPTIMIZER: {"module": {DEEP_CONFIG_DTYPE: str,
                                       DEEP_CONFIG_DEFAULT: None},
                                       "name": {DEEP_CONFIG_DTYPE: str,
                                                DEEP_CONFIG_DEFAULT: "Adam"},
                                       "kwargs": {DEEP_CONFIG_DTYPE: dict,
                                                  DEEP_CONFIG_DEFAULT: {"lr": 0.001,
                                                                        "eps": 0.000000001,
                                                                        "amsgrad": False,
                                                                        "betas": [0.9, 0.999],
                                                                        "weight_decay": 0.0}
                                                  }
                                       },
               DEEP_CONFIG_HISTORY: {"verbose": {DEEP_CONFIG_DTYPE: int,
                                                 DEEP_CONFIG_DEFAULT: 1},
                                     "memorize": {DEEP_CONFIG_DTYPE: int,
                                                  DEEP_CONFIG_DEFAULT: 1}},
               DEEP_CONFIG_TRAINING: {"num_epochs": {DEEP_CONFIG_DTYPE: int,
                                                     DEEP_CONFIG_DEFAULT: 10},
                                      "initial_epoch": {DEEP_CONFIG_DTYPE: int,
                                                        DEEP_CONFIG_DEFAULT: 0},
                                      "shuffle": {DEEP_CONFIG_DTYPE: str,
                                                  DEEP_CONFIG_DEFAULT: "None"},
                                      "save_condition": {DEEP_CONFIG_DTYPE: int,
                                                         DEEP_CONFIG_DEFAULT: 1},
                                      "save_method": {DEEP_CONFIG_DTYPE: int,
                                                      DEEP_CONFIG_DEFAULT: 1}
                                      },
               DEEP_CONFIG_DATA: {"dataloader": {"batch_size": {DEEP_CONFIG_DTYPE: int,
                                                                DEEP_CONFIG_DEFAULT: 32},
                                                 "num_workers": {DEEP_CONFIG_DTYPE: int,
                                                                 DEEP_CONFIG_DEFAULT: 1}
                                                 },
                                  DEEP_CONFIG_ENABLED: {"train": {DEEP_CONFIG_DTYPE: bool,
                                                                  DEEP_CONFIG_DEFAULT: True},
                                                        "validation": {DEEP_CONFIG_DTYPE: bool,
                                                                       DEEP_CONFIG_DEFAULT: True},
                                                        "test": {DEEP_CONFIG_DTYPE: bool,
                                                                 DEEP_CONFIG_DEFAULT: True}
                                                        },
                                  "dataset": {"train": {"inputs": {DEEP_CONFIG_DTYPE: [{"source": [str],
                                                                                        "join": [str],
                                                                                        "type": str,
                                                                                        "load_method": str}],
                                                                   DEEP_CONFIG_DEFAULT: None},
                                                        "labels": {DEEP_CONFIG_DTYPE: [{"source": [str],
                                                                                        "join": [str],
                                                                                        "type": str,
                                                                                        "load_method": str}],
                                                                   DEEP_CONFIG_DEFAULT: None},
                                                        "additional_data": {DEEP_CONFIG_DTYPE: [{"source": [str],
                                                                                                 "join": [str],
                                                                                                 "type": str,
                                                                                                 "load_method": str}],
                                                                            DEEP_CONFIG_DEFAULT: None},
                                                        "number": {DEEP_CONFIG_DTYPE: int,
                                                                   DEEP_CONFIG_DEFAULT: None},
                                                        "name": {DEEP_CONFIG_DTYPE: str,
                                                                 DEEP_CONFIG_DEFAULT: "train"}
                                                        },
                                              "validation": {"inputs": {DEEP_CONFIG_DTYPE: [{"source": [str],
                                                                                             "join": [str],
                                                                                             "type": str,
                                                                                             "load_method": str}],
                                                                        DEEP_CONFIG_DEFAULT: None},
                                                             "labels": {DEEP_CONFIG_DTYPE: [{"source": [str],
                                                                                             "join": [str],
                                                                                             "type": str,
                                                                                             "load_method": str}],
                                                                        DEEP_CONFIG_DEFAULT: None},
                                                             "additional_data": {DEEP_CONFIG_DTYPE: [{"source": [str],
                                                                                                      "join": [str],
                                                                                                      "type": str,
                                                                                                      "load_method": str}],
                                                                                 DEEP_CONFIG_DEFAULT: None},
                                                             "number": {DEEP_CONFIG_DTYPE: int,
                                                                        DEEP_CONFIG_DEFAULT: None},
                                                             "name": {DEEP_CONFIG_DTYPE: str,
                                                                      DEEP_CONFIG_DEFAULT: "validation"}
                                                             },
                                              "test": {"inputs": {DEEP_CONFIG_DTYPE: [{"source": [str],
                                                                                       "join": [str],
                                                                                       "type": str,
                                                                                       "load_method": str}],
                                                                  DEEP_CONFIG_DEFAULT: None},
                                                       "labels": {DEEP_CONFIG_DTYPE: [{"source": [str],
                                                                                       "join": [str],
                                                                                       "type": str,
                                                                                       "load_method": str}],
                                                                  DEEP_CONFIG_DEFAULT: None},
                                                       "additional_data": {DEEP_CONFIG_DTYPE: [{"source": [str],
                                                                                                "join": [str],
                                                                                                "type": str,
                                                                                                "load_method": str}],
                                                                           DEEP_CONFIG_DEFAULT: None},
                                                       "number": {DEEP_CONFIG_DTYPE: int,
                                                                  DEEP_CONFIG_DEFAULT: None},
                                                       "name": {DEEP_CONFIG_DTYPE: str,
                                                                DEEP_CONFIG_DEFAULT: "test"}
                                                       }
                                              }
                                  },
               DEEP_CONFIG_LOSS: {DEEP_CONFIG_WILDCARD: {"module": {DEEP_CONFIG_DTYPE: str,
                                                         DEEP_CONFIG_DEFAULT: None},
                                                         "name": {DEEP_CONFIG_DTYPE: str,
                                                                  DEEP_CONFIG_DEFAULT: "CrossEntropyLoss"},
                                                         "weight": {DEEP_CONFIG_DTYPE: float,
                                                                    DEEP_CONFIG_DEFAULT: 1},
                                                         "kwargs": {DEEP_CONFIG_DTYPE: dict,
                                                                    DEEP_CONFIG_DEFAULT: {}}
                                                         }
                                  },
               DEEP_CONFIG_TRANSFORM: {},
               DEEP_CONFIG_METRICS: {DEEP_CONFIG_WILDCARD: {"module": {DEEP_CONFIG_DTYPE: str,
                                                                       DEEP_CONFIG_DEFAULT: None},
                                                            "name": {DEEP_CONFIG_DTYPE: str,
                                                                     DEEP_CONFIG_DEFAULT: "CrossEntropyLoss"},
                                                            "weight": {DEEP_CONFIG_DTYPE: float,
                                                                       DEEP_CONFIG_DEFAULT: 1},
                                                            "kwargs": {DEEP_CONFIG_DTYPE: dict,
                                                                       DEEP_CONFIG_DEFAULT: {}}
                                                            }
                                     }
               }

# A dict of names for each config file
DEEP_CONFIG_FILES = {item: "%s%s" % (item, DEEP_EXT_YAML) for item in DEEP_CONFIG_SECTIONS}
