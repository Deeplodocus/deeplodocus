"""
Namespace object for storing complex, multi-level data.
Data can added on initialisation as a dictionary, a path to a directory and/or a path to a yaml_file
"""

import yaml


class Namespace(object):

    def __init__(self, *args):
        """
        :param args: str or dict: yaml path or dictionary to be included in the namespace
        """
        for arg in args:
            if isinstance(arg, str):
                for key, item in self.__yaml2namespace(arg).get().items():
                    self.add({key: item})
            elif isinstance(arg, dict):
                for key, item in self.__dict2namespace(arg).get().items():
                    self.add({key: item})
            else:
                print("Warning: unused argument %s" % arg)

    def add(self, dictionary, sub_space=None):
        """
        Add a new dictionary to the namespace.
        :param dictionary: dict: the dictionary of data to add.
        :param sub_space: str or list of str: path to a sub-space within the Namespace.
        :return: None.
        """
        if sub_space is None:
            self.__dict__.update(dictionary)
        else:
            if isinstance(sub_space, list):
                sub_space = sub_space[0] if len(sub_space) == 1 else sub_space
            if isinstance(sub_space, list):
                if sub_space[0] not in self.get().keys():
                    self.__dict__.update({sub_space[0]: Namespace()})
                try:
                    self.get()[sub_space[0]].add(dictionary, sub_space[1:])
                except AttributeError:
                    raise AttributeError("Unable to add data to '%s', location not a namespace " % sub_space[0])
            else:
                if sub_space not in self.get().keys():
                    self.__dict__.update({sub_space: Namespace()})
                self.get()[sub_space].add(dictionary)

    def get(self, key=None):
        """
        Get data from the Namespace as a dictionary given a key or list of keys.
        Note: key can be a list of sub Namespace names for which get() will be called recursively.
        :param key: str or list of str: key of the sub item to be returned.
        :return: dict: data retrieved from the Namespace.
        """
        if key is None:
            return self.__dict__
        else:
            if isinstance(key, list):
                key = key[0] if len(key) == 1 else key
            if isinstance(key, list):
                return self.get(key[0]).get(key=key[1:])
            else:
                return self.__dict__[key]

    def save(self, file_name, tab_size=2):
        """
        :param tab_size:
        :return:
        """
        with open(file_name, "w") as file:
            file.write(self.get_summary(tab_size=tab_size))

    def get_summary(self, tab_size=2, tabs=0, line=None):
        """
        Prints a summary of all the data in the Namespace.
        :param tab_size: int: number of spaces per tab.
        :param tabs: int: number of tabs to use (for internal use).
        :param line: str: the summary statement from a previous call (for internal use)
        :return: None.
        """
        if line is None:
            line = ""
        for key, item in self.__dict__.items():
            if isinstance(item, Namespace):
                line += "%s%s:\n" % (" " * tab_size * tabs, key)
                line += item.get_summary(tabs=tabs + 1)
            else:
                line += "%s%s: %s\n" % (" " * tab_size * tabs, key, item)
        return line

    def summary(self, tab_size=2):
        """
        :param tab_size:
        :return:
        """
        print(self.get_summary(tab_size=tab_size))

    def __dict2namespace(self, dictionary):
        """
        Given a dictionary, converts it into a Namespace object.
        :param dictionary: dict: a dictionary containing information to store as a Namespace.
        :return: Namespace containing all data from the dictionary.
        Note: the function is recursive and sub-dictionaries are represented as sub-Namespaces.
        """
        namespace = Namespace()
        for key, item in dictionary.items():
            if isinstance(item, dict):
                item = self.__dict2namespace(item)
            namespace.add({key: item})
        return namespace

    def __yaml2namespace(self, file_name):
        """
        Given a yaml file, opens it and converts it to a Namespace object.
        :param file_name: str: path to a yaml file.
        :return: Namespace containing all data from the yaml file.
        """
        with open(file_name, "r") as file:
            dictionary = yaml.load(file)
            return self.__dict2namespace(dictionary)


if __name__ == "__main__":
    namespace = Namespace({"a": 1, "b": 2, "c": {"d": 5, "e": 6}})
    a = {"project": "../yaml1", "network": "yaml2"}
    namespace.add(a, ["c", "main", "moop"])
    namespace.save("/home/samuel/config.yaml")

