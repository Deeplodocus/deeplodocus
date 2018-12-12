

class Flag(object):


    def __init__(self, index : int, name : str, description: str =  ""):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------
        Initialize a new flag.
        Each new flag contains a unique index and a corresponding name.
        The description is optional and not recommended for memory efficiency

        PARAMETERS:
        -----------

        :param index (int): The index of the flag
        :param name (str): Name of the flag
        :param description (str, Optional): Description of the flag

        RETURN:
        -------

        None
        """
        self.index = index
        self.name = name
        self.description = description

    def __call__(self, *args, **kwargs):
        return self.index

    def __str__(self):
        return "({0} ({1}))".format(self.x,self.y)

    def get_index(self):
        return self.index

    def get_name(self):
        return self.name

    def get_description(self):
        return self.description


DEEP_LIB_OPENCV = Flag(0, "OpenCV Library")

a = DEEP_LIB_OPENCV
print(a)