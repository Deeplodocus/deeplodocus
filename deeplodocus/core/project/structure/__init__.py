"""
Use the DEEP_DIRECTORY_TREE to define the structure of a new
deeplodocus project (excluding the config dir).

Each key will result in the creation of a new directory, unless the key == FILES
If a directory should be empty, let its value be None
If a directory contains either or some files, let its value be a dict with the
structure:
{
    sub-directory-1: None
    sub-directory-2: None
    FILES: [file-1.txt, file-2.txt]
}

Files must be given in a list, and the root of
files referenced here is deeplodocus/core/project/structure
"""

DEEP_DIRECTORY_TREE = {
    "modules": {
        "models": None,
        "sources": None,
        "losses": None,
        "metrics": None,
        "transforms": None,
        "optimizers": None
    },
    "data": None,
    "config": {
        "transformers": {
            "FILES": ["input.yaml"]
        },
        "FILES": ["data.yaml"]
    },
    "FILES": ["main.py"]
}
