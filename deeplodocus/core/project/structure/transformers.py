# Keywords
DEEP_CONFIG_INIT = "INIT"
DEEP_CONFIG_DEFAULT = "DEFAULT"
DEEP_CONFIG_DTYPE = "DTYPE"
DEEP_CONFIG_WILDCARD = "*"
DEEP_CONFIG_COMMENT = "COMMENT"

OUTPUT_TRANSFORMER = {
    DEEP_CONFIG_COMMENT: "# For more information about output transformers visit www.deeplodocus.org\n",
    "name": {
        DEEP_CONFIG_DEFAULT: "Output Transformer",
        DEEP_CONFIG_DTYPE: str,
        DEEP_CONFIG_INIT: "Output Transformer"
    },
    "transforms": {
        DEEP_CONFIG_WILDCARD: {
            DEEP_CONFIG_INIT: "UniqueName",
            "name": {
                DEEP_CONFIG_DEFAULT: None,
                DEEP_CONFIG_DTYPE: str,
                DEEP_CONFIG_INIT: "     # The name of your transform class or function goes here"
            },
            "module": {
                DEEP_CONFIG_DEFAULT: None,
                DEEP_CONFIG_DTYPE: str,
                DEEP_CONFIG_INIT: "   # The module that contains your transform goes here"
            },
            "kwargs": {
                DEEP_CONFIG_DEFAULT: None,
                DEEP_CONFIG_DTYPE: str,
                DEEP_CONFIG_INIT: "{} # Specify any arguments here, keep as {} if there are no arguments"
            }
        }
    }
}

ONEOF_TRANSFORMER = {
    DEEP_CONFIG_COMMENT + "0": "# For more information about input transformers visit www.deeplodocus.org\n",
    "method": {
        DEEP_CONFIG_DEFAULT: "oneof",
        DEEP_CONFIG_DTYPE: str,
    },
    "name": {
        DEEP_CONFIG_DEFAULT: "One-of Transformer",
        DEEP_CONFIG_DTYPE: str,
    },
    DEEP_CONFIG_COMMENT + "1": "\n# Every transform in here will be called sequentially at the start:",
    "mandatory_transform_start": [
        {
            DEEP_CONFIG_WILDCARD: {
                DEEP_CONFIG_INIT: "<Transform-Name>",
                "name": {
                    DEEP_CONFIG_DEFAULT: None,
                    DEEP_CONFIG_DTYPE: str,
                    DEEP_CONFIG_INIT: "     # The name of your transform class or function goes here"
                },
                "module": {
                    DEEP_CONFIG_DEFAULT: None,
                    DEEP_CONFIG_DTYPE: str,
                    DEEP_CONFIG_INIT: "   # The module that contains your transform goes here"
                },
                "kwargs": {
                    DEEP_CONFIG_DEFAULT: None,
                    DEEP_CONFIG_DTYPE: str,
                    DEEP_CONFIG_INIT: "{} # Specify any arguments here, keep as {} if there are no arguments"
                }
            }
        }
    ],
    DEEP_CONFIG_COMMENT + "2": "\n# Then one and only one transform will be selected at random from here:",
    "transforms": [
        {
            DEEP_CONFIG_WILDCARD: {
                DEEP_CONFIG_INIT: "<Transform-Name>",
                "name": {
                    DEEP_CONFIG_DEFAULT: None,
                    DEEP_CONFIG_DTYPE: str,
                    DEEP_CONFIG_INIT: "     # The name of your transform class or function goes here"
                },
                "module": {
                    DEEP_CONFIG_DEFAULT: None,
                    DEEP_CONFIG_DTYPE: str,
                    DEEP_CONFIG_INIT: "   # The module that contains your transform goes here"
                },
                "kwargs": {
                    DEEP_CONFIG_DEFAULT: None,
                    DEEP_CONFIG_DTYPE: str,
                    DEEP_CONFIG_INIT: "{} # Specify any arguments here, keep as {} if there are no arguments"
                }
            }
        }
    ],
    DEEP_CONFIG_COMMENT + "3": "\n# Every transform in here will be called sequentially at the end:",
    "mandatory_transforms_end": [
        {
            DEEP_CONFIG_WILDCARD: {
                DEEP_CONFIG_INIT: "<Transform-Name>",
                "name": {
                    DEEP_CONFIG_DEFAULT: None,
                    DEEP_CONFIG_DTYPE: str,
                    DEEP_CONFIG_INIT: "     # The name of your transform class or function goes here"
                },
                "module": {
                    DEEP_CONFIG_DEFAULT: None,
                    DEEP_CONFIG_DTYPE: str,
                    DEEP_CONFIG_INIT: "   # The module that contains your transform goes here"
                },
                "kwargs": {
                    DEEP_CONFIG_DEFAULT: None,
                    DEEP_CONFIG_DTYPE: str,
                    DEEP_CONFIG_INIT: "{} # Specify any arguments here, keep as {} if there are no arguments"
                }
            }
        }
    ],
}

SEQUENTIAL_TRANSFORMER = {
    DEEP_CONFIG_COMMENT + "0": "# For more information about input transformers visit www.deeplodocus.org\n",
    "method": {
        DEEP_CONFIG_DEFAULT: "sequential",
        DEEP_CONFIG_DTYPE: str,
    },
    "name": {
        DEEP_CONFIG_DEFAULT: "Sequential Transformer",
        DEEP_CONFIG_DTYPE: str,
    },
    DEEP_CONFIG_COMMENT + "1": "\n# Every transform in here will be called sequentially at the start:",
    "mandatory_transform_start": [
        {
            DEEP_CONFIG_WILDCARD: {
                DEEP_CONFIG_INIT: "<Transform-Name>",
                "name": {
                    DEEP_CONFIG_DEFAULT: None,
                    DEEP_CONFIG_DTYPE: str,
                    DEEP_CONFIG_INIT: "     # The name of your transform class or function goes here"
                },
                "module": {
                    DEEP_CONFIG_DEFAULT: None,
                    DEEP_CONFIG_DTYPE: str,
                    DEEP_CONFIG_INIT: "   # The module that contains your transform goes here"
                },
                "kwargs": {
                    DEEP_CONFIG_DEFAULT: None,
                    DEEP_CONFIG_DTYPE: str,
                    DEEP_CONFIG_INIT: "{} # Specify any arguments here, keep as {} if there are no arguments"
                }
            }
        }
    ],
    DEEP_CONFIG_COMMENT + "2": "\n# Then every transform in here will be called sequentially:",
    "transforms": [
        {
            DEEP_CONFIG_WILDCARD: {
                DEEP_CONFIG_INIT: "<Transform-Name>",
                "name": {
                    DEEP_CONFIG_DEFAULT: None,
                    DEEP_CONFIG_DTYPE: str,
                    DEEP_CONFIG_INIT: "     # The name of your transform class or function goes here"
                },
                "module": {
                    DEEP_CONFIG_DEFAULT: None,
                    DEEP_CONFIG_DTYPE: str,
                    DEEP_CONFIG_INIT: "   # The module that contains your transform goes here"
                },
                "kwargs": {
                    DEEP_CONFIG_DEFAULT: None,
                    DEEP_CONFIG_DTYPE: str,
                    DEEP_CONFIG_INIT: "{} # Specify any arguments here, keep as {} if there are no arguments"
                }
            }
        }
    ],
    DEEP_CONFIG_COMMENT + "3": "\n# Every transform in here will be called sequentially at the end:",
    "mandatory_transforms_end": [
        {
            DEEP_CONFIG_WILDCARD: {
                DEEP_CONFIG_INIT: "<Transform-Name>",
                "name": {
                    DEEP_CONFIG_DEFAULT: None,
                    DEEP_CONFIG_DTYPE: str,
                    DEEP_CONFIG_INIT: "     # The name of your transform class or function goes here"
                },
                "module": {
                    DEEP_CONFIG_DEFAULT: None,
                    DEEP_CONFIG_DTYPE: str,
                    DEEP_CONFIG_INIT: "   # The module that contains your transform goes here"
                },
                "kwargs": {
                    DEEP_CONFIG_DEFAULT: None,
                    DEEP_CONFIG_DTYPE: str,
                    DEEP_CONFIG_INIT: "{} # Specify any arguments here, keep as {} if there are no arguments"
                }
            }
        }
    ],
}


SOMEOF_TRANSFORMER = {
    DEEP_CONFIG_COMMENT + "0": "# For more information about input transformers visit www.deeplodocus.org\n",
    "method": {
        DEEP_CONFIG_DEFAULT: "someof",
        DEEP_CONFIG_DTYPE: str,
    },
    "name": {
        DEEP_CONFIG_DEFAULT: "Some-of Transformer",
        DEEP_CONFIG_DTYPE: str,
    },
    "num_transforms": {
        DEEP_CONFIG_DEFAULT: 1,
        DEEP_CONFIG_DTYPE: int,
        DEEP_CONFIG_INIT: "1        # The number of transforms to select"
    },
    "num_transofrms_min": {
        DEEP_CONFIG_DEFAULT: None,
        DEEP_CONFIG_DTYPE: int,
        DEEP_CONFIG_INIT: "Null # Minimum number of transforms to use (only used if num_transforms is not given)"
    },
    "num_transforms_max": {
        DEEP_CONFIG_DEFAULT: None,
        DEEP_CONFIG_DTYPE: int,
        DEEP_CONFIG_INIT: "Null # Maximum number of transforms to use (only used if num_transforms it not given)"
    },
    DEEP_CONFIG_COMMENT + "1": "\n# Every transform in here will be called sequentially at the start:",
    "mandatory_transform_start": [
        {
            DEEP_CONFIG_WILDCARD: {
                DEEP_CONFIG_INIT: "<Transform-Name>",
                "name": {
                    DEEP_CONFIG_DEFAULT: None,
                    DEEP_CONFIG_DTYPE: str,
                    DEEP_CONFIG_INIT: "     # The name of your transform class or function goes here"
                },
                "module": {
                    DEEP_CONFIG_DEFAULT: None,
                    DEEP_CONFIG_DTYPE: str,
                    DEEP_CONFIG_INIT: "   # The module that contains your transform goes here"
                },
                "kwargs": {
                    DEEP_CONFIG_DEFAULT: None,
                    DEEP_CONFIG_DTYPE: str,
                    DEEP_CONFIG_INIT: "{} # Specify any arguments here, keep as {} if there are no arguments"
                }
            }
        }
    ],
    DEEP_CONFIG_COMMENT + "2": "\n# Then some of the transforms in here will be called:",
    "transforms": [
        {
            DEEP_CONFIG_WILDCARD: {
                DEEP_CONFIG_INIT: "<Transform-Name>",
                "name": {
                    DEEP_CONFIG_DEFAULT: None,
                    DEEP_CONFIG_DTYPE: str,
                    DEEP_CONFIG_INIT: "     # The name of your transform class or function goes here"
                },
                "module": {
                    DEEP_CONFIG_DEFAULT: None,
                    DEEP_CONFIG_DTYPE: str,
                    DEEP_CONFIG_INIT: "   # The module that contains your transform goes here"
                },
                "kwargs": {
                    DEEP_CONFIG_DEFAULT: None,
                    DEEP_CONFIG_DTYPE: str,
                    DEEP_CONFIG_INIT: "{} # Specify any arguments here, keep as {} if there are no arguments"
                }
            }
        }
    ],
    DEEP_CONFIG_COMMENT + "3": "\n# Every transform in here will be called sequentially at the end:",
    "mandatory_transforms_end": [
        {
            DEEP_CONFIG_WILDCARD: {
                DEEP_CONFIG_INIT: "<Transform-Name>",
                "name": {
                    DEEP_CONFIG_DEFAULT: None,
                    DEEP_CONFIG_DTYPE: str,
                    DEEP_CONFIG_INIT: "     # The name of your transform class or function goes here"
                },
                "module": {
                    DEEP_CONFIG_DEFAULT: None,
                    DEEP_CONFIG_DTYPE: str,
                    DEEP_CONFIG_INIT: "   # The module that contains your transform goes here"
                },
                "kwargs": {
                    DEEP_CONFIG_DEFAULT: None,
                    DEEP_CONFIG_DTYPE: str,
                    DEEP_CONFIG_INIT: "{} # Specify any arguments here, keep as {} if there are no arguments"
                }
            }
        }
    ],
}
