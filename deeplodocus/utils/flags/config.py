from deeplodocus.utils.flags.ext import DEEP_EXT_YAML

# The divider to use when expressing paths to configs
DEEP_CONFIG_DIVIDER = "/"

# Keywords
DEEP_CONFIG_INIT = "INIT"
DEEP_CONFIG_DEFAULT = "DEFAULT"
DEEP_CONFIG_DTYPE = "DTYPE"
DEEP_CONFIG_WILDCARD = "*"

# Names of each section of the config fle
DEEP_CONFIG_PROJECT = "project"
DEEP_CONFIG_MODEL = "model"
DEEP_CONFIG_TRAINING = "training"
DEEP_CONFIG_DATA = "data"
DEEP_CONFIG_TRANSFORM = "transform"
DEEP_CONFIG_OPTIMIZER = "optimizer"
DEEP_CONFIG_METRICS = "metrics"
DEEP_CONFIG_LOSSES = "losses"
DEEP_CONFIG_HISTORY = "history"

# Wildcard place holders
DEEP_CONFIG_WILDCARD_DEFAULT = {DEEP_CONFIG_METRICS: "accuracy",
                                DEEP_CONFIG_LOSSES: "loss"}

# List of all config sections
DEEP_CONFIG_SECTIONS = [DEEP_CONFIG_PROJECT,
                        DEEP_CONFIG_MODEL,
                        DEEP_CONFIG_TRAINING,
                        DEEP_CONFIG_DATA,
                        DEEP_CONFIG_TRANSFORM,
                        DEEP_CONFIG_OPTIMIZER,
                        DEEP_CONFIG_METRICS,
                        DEEP_CONFIG_LOSSES,
                        DEEP_CONFIG_HISTORY]

# A dict of names for each config file
DEEP_CONFIG_FILES = {item: "%s%s" % (item, DEEP_EXT_YAML) for item in DEEP_CONFIG_SECTIONS}

# Define the expected structure of the project configuration space
# First level keys define the name of config .yaml files
# Second level keys and below should be found in each file
# Default values and  data types must be given for each configuration
# NB: if a list of floats is expected, use [float] instead of float
DEEP_CONFIG = {
    DEEP_CONFIG_PROJECT: {
        "sub_project": {
            DEEP_CONFIG_DTYPE: str,
            DEEP_CONFIG_DEFAULT: "version01"
        },
        "cv_library": {
            DEEP_CONFIG_DTYPE: str,
            DEEP_CONFIG_DEFAULT: "pil"
        },
        "device": {
            DEEP_CONFIG_DTYPE: str,
            DEEP_CONFIG_DEFAULT: "auto"
        },
        "device_ids": {
            DEEP_CONFIG_DTYPE: [int],
            DEEP_CONFIG_DEFAULT: "auto"
        },
        "logs": {
            "history_train_batches": {
                DEEP_CONFIG_DTYPE: bool,
                DEEP_CONFIG_DEFAULT: True
            },
            "history_train_epochs": {
                DEEP_CONFIG_DTYPE: bool,
                DEEP_CONFIG_DEFAULT: True
            },
            "history_validation": {
                DEEP_CONFIG_DTYPE: bool,
                DEEP_CONFIG_DEFAULT: True
            },
            "notification": {
                DEEP_CONFIG_DTYPE: bool,
                DEEP_CONFIG_DEFAULT: True
            }
        },
        "on_wake": {
            DEEP_CONFIG_DTYPE: [str],
            DEEP_CONFIG_DEFAULT: None
        }
    },
    DEEP_CONFIG_MODEL: {
        "from_file": {
            DEEP_CONFIG_DTYPE: bool,
            DEEP_CONFIG_DEFAULT: False
        },
        "module": {
            DEEP_CONFIG_DTYPE: str,
            DEEP_CONFIG_DEFAULT: None,
            DEEP_CONFIG_INIT: "torchvision.models"
        },
        "name": {
            DEEP_CONFIG_DTYPE: str,
            DEEP_CONFIG_DEFAULT: "vgg16_bn"
        },
        "input_size": {
            DEEP_CONFIG_DTYPE: [[int]],
            DEEP_CONFIG_DEFAULT: None,
            DEEP_CONFIG_INIT: [[3, 224, 224]]
        },
        "kwargs": {
            DEEP_CONFIG_DTYPE: dict,
            DEEP_CONFIG_DEFAULT: {},
            DEEP_CONFIG_INIT: {
                "num_classes": 1000,
                "pretrained": True
            }
        }
    },
    DEEP_CONFIG_OPTIMIZER: {
        "module": {
            DEEP_CONFIG_DTYPE: str,
            DEEP_CONFIG_DEFAULT: None,
            DEEP_CONFIG_INIT: "torch.optim"
        },
        "name": {
            DEEP_CONFIG_DTYPE: str,
            DEEP_CONFIG_DEFAULT: "Adam"
        },
        "kwargs": {
            DEEP_CONFIG_DTYPE: dict,
            DEEP_CONFIG_DEFAULT: {
                "lr": 0.001,
                "eps": 0.000000001,
                "amsgrad": False,
                "betas": [0.9, 0.999],
                "weight_decay": 0.0
            }
        }
    },
    DEEP_CONFIG_HISTORY: {
        "verbose": {
            DEEP_CONFIG_DTYPE: str,
            DEEP_CONFIG_DEFAULT: "default"
        },
        "memorize": {
            DEEP_CONFIG_DTYPE: str,
            DEEP_CONFIG_DEFAULT: "batch"
        }
    },
    DEEP_CONFIG_TRAINING: {
        "num_epochs": {
            DEEP_CONFIG_DTYPE: int,
            DEEP_CONFIG_DEFAULT: 10
        },
        "initial_epoch": {
            DEEP_CONFIG_DTYPE: int,
            DEEP_CONFIG_DEFAULT: 0
        },
        "shuffle": {
            DEEP_CONFIG_DTYPE: str,
            DEEP_CONFIG_DEFAULT: "default"
        },
        "saver": {
            "method": {
                DEEP_CONFIG_DEFAULT: "pytorch",
                DEEP_CONFIG_DTYPE: str
            },
            "save_signal": {
                DEEP_CONFIG_DEFAULT: "auto",
                DEEP_CONFIG_DTYPE: str
            },
            "overwrite": {
                DEEP_CONFIG_DEFAULT: False,
                DEEP_CONFIG_DTYPE: bool
            }
        },
        "overwatch": {
            "name": {
                DEEP_CONFIG_DEFAULT: "Total Loss",
                DEEP_CONFIG_DTYPE: str
            },
            "condition": {
                DEEP_CONFIG_DEFAULT: "less",
                DEEP_CONFIG_DTYPE: str
            }
        },
    },
    DEEP_CONFIG_DATA: {
        "dataloader": {
            "batch_size": {
                DEEP_CONFIG_DTYPE: int,
                DEEP_CONFIG_DEFAULT: 32
            },
            "num_workers": {
                DEEP_CONFIG_DTYPE: int,
                DEEP_CONFIG_DEFAULT: 1
            }
        },
        "enabled": {
            "train": {
                DEEP_CONFIG_DTYPE: bool,
                DEEP_CONFIG_DEFAULT: True
            },
            "validation": {
                DEEP_CONFIG_DTYPE: bool,
                DEEP_CONFIG_DEFAULT: True
            },
            "test": {
                DEEP_CONFIG_DTYPE: bool,
                DEEP_CONFIG_DEFAULT: True
            }
        },
        "dataset": {
            "train": {
                "inputs": {
                    DEEP_CONFIG_DTYPE: [{
                        "source": [str],
                        "join": [str],
                        "type": str,
                        "load_method": str
                    }],
                    DEEP_CONFIG_DEFAULT: None
                },
                "labels": {
                    DEEP_CONFIG_DTYPE: [{
                        "source": [str],
                        "join": [str],
                        "type": str,
                        "load_method": str
                    }],
                    DEEP_CONFIG_DEFAULT: None
                },
                "additional_data": {
                    DEEP_CONFIG_DTYPE: [{
                        "source": [str],
                        "join": [str],
                        "type": str,
                        "load_method": str
                    }],
                    DEEP_CONFIG_DEFAULT: None
                },
                "number": {
                    DEEP_CONFIG_DTYPE: int,
                    DEEP_CONFIG_DEFAULT: None
                },
                "name": {
                    DEEP_CONFIG_DTYPE: str,
                    DEEP_CONFIG_DEFAULT: "Training"
                }
            },
            "validation": {
                "inputs": {
                    DEEP_CONFIG_DTYPE: [{
                        "source": [str],
                        "join": [str],
                        "type": str,
                        "load_method": str
                    }],
                    DEEP_CONFIG_DEFAULT: None
                },
                "labels": {DEEP_CONFIG_DTYPE: [{
                    "source": [str],
                    "join": [str],
                    "type": str,
                    "load_method": str
                }],
                    DEEP_CONFIG_DEFAULT: None
                },
                "additional_data": {
                    DEEP_CONFIG_DTYPE: [{
                        "source": [str],
                        "join": [str],
                        "type": str,
                        "load_method": str}],
                    DEEP_CONFIG_DEFAULT: None
                },
                "number": {
                    DEEP_CONFIG_DTYPE: int,
                    DEEP_CONFIG_DEFAULT: None
                },
                "name": {
                    DEEP_CONFIG_DTYPE: str,
                    DEEP_CONFIG_DEFAULT: "Validation"
                }
            },
            "test": {
                "inputs": {
                    DEEP_CONFIG_DTYPE: [{
                        "source": [str],
                        "join": [str],
                        "type": str,
                        "load_method": str
                    }],
                    DEEP_CONFIG_DEFAULT: None
                },
                "labels": {
                    DEEP_CONFIG_DTYPE: [{
                        "source": [str],
                        "join": [str],
                        "type": str,
                        "load_method": str
                    }],
                    DEEP_CONFIG_DEFAULT: None
                },
                "additional_data": {
                    DEEP_CONFIG_DTYPE: [{
                        "source": [str],
                        "join": [str],
                        "type": str,
                        "load_method": str
                    }],
                    DEEP_CONFIG_DEFAULT: None
                },
                "number": {
                    DEEP_CONFIG_DTYPE: int,
                    DEEP_CONFIG_DEFAULT: None
                },
                "name": {
                    DEEP_CONFIG_DTYPE: str,
                    DEEP_CONFIG_DEFAULT: "Test"
                }
            }
        }
    },
    DEEP_CONFIG_LOSSES: {
        DEEP_CONFIG_WILDCARD: {
            "module": {
                DEEP_CONFIG_DTYPE: str,
                DEEP_CONFIG_DEFAULT: None
            },
            "name": {
                DEEP_CONFIG_DTYPE: str,
                DEEP_CONFIG_DEFAULT: "CrossEntropyLoss"
            },
            "weight": {
                DEEP_CONFIG_DTYPE: float,
                DEEP_CONFIG_DEFAULT: 1
            },
            "kwargs": {
                DEEP_CONFIG_DTYPE: dict,
                DEEP_CONFIG_DEFAULT: {}
            }
        }
    },
    DEEP_CONFIG_TRANSFORM: {
        "train": {
            "name": {
                DEEP_CONFIG_DEFAULT: "Train Transform Manager",
                DEEP_CONFIG_DTYPE: str
            },
            "inputs": {
                DEEP_CONFIG_DEFAULT: None,
                DEEP_CONFIG_DTYPE: [str]
            },
            "labels": {
                DEEP_CONFIG_DEFAULT: None,
                DEEP_CONFIG_DTYPE: [str]
            },
            "additional_data": {
                DEEP_CONFIG_DEFAULT: None,
                DEEP_CONFIG_DTYPE: [str]
            }
        },
        "validation": {
            "name": {
                DEEP_CONFIG_DEFAULT: "Validation Transform Manager",
                DEEP_CONFIG_DTYPE: str
            },
            "inputs": {
                DEEP_CONFIG_DEFAULT: None,
                DEEP_CONFIG_DTYPE: [str]
            },
            "labels": {
                DEEP_CONFIG_DEFAULT: None,
                DEEP_CONFIG_DTYPE: [str]
            },
            "additional_data": {
                DEEP_CONFIG_DEFAULT: None,
                DEEP_CONFIG_DTYPE: [str]
            }
        },
        "test": {
            "name": {
                DEEP_CONFIG_DEFAULT: "Test Transform Manager",
                DEEP_CONFIG_DTYPE: str
            },
            "inputs": {
                DEEP_CONFIG_DEFAULT: None,
                DEEP_CONFIG_DTYPE: [str]
            },
            "labels": {
                DEEP_CONFIG_DEFAULT: None,
                DEEP_CONFIG_DTYPE: [str]
            },
            "additional_data":
                {
                    DEEP_CONFIG_DEFAULT: None,
                    DEEP_CONFIG_DTYPE: [str]
                }
        }
    },
    DEEP_CONFIG_METRICS: {
        DEEP_CONFIG_WILDCARD: {
            "module": {
                DEEP_CONFIG_DTYPE: str,
                DEEP_CONFIG_DEFAULT: None
            },
            "name": {
                DEEP_CONFIG_DTYPE: str,
                DEEP_CONFIG_DEFAULT: "accuracy"
            },
            "weight": {DEEP_CONFIG_DTYPE: float,
                       DEEP_CONFIG_DEFAULT: 1
                       },
            "kwargs": {
                DEEP_CONFIG_DTYPE: dict,
                DEEP_CONFIG_DEFAULT: {}
            }
        }
    }
}
