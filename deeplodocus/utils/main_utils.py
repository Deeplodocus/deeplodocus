import os
import __main__

def get_main_path():
    return os.path.dirname(os.path.abspath(__main__.__file__))