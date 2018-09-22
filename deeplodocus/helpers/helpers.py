#!/usr/bin/env python3

import os
import yaml


def nested_set(dic, keys, value):
    """
    :param dic:
    :param keys:
    :param value:
    :return:
    """
    dictionary = dic
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value
    return dictionary


def print_dict(dictionary, indent=2, level=0):
    """
    :param dictionary:
    :param indent:
    :param level:
    :return:
    """
    space = " " * indent
    for key, item in dictionary.items():
        if isinstance(item, dict):
            print("%s%s:" % (space * level, key))
            print_dict(item, indent=indent, level=level+1)
        elif isinstance(item, list):
            print("%s%s:" % (space * level, key))
            for i in item:
                if isinstance(i, dict):
                    print_dict(i, indent=indent, level=level+1)
                else:
                    print("%s-%s" % (space * (level + 1), i))
        else:
            print("%s%s: %s" % (space * level, key, str(item)))


def save_dict(dictionary, path, indent=4, level=0):
    """
    :param dictionary:
    :param path:
    :param indent:
    :param level:
    :return:
    """
    space = " " * indent * level
    for key, item in dictionary.items():
        if isinstance(item, dict):
            with open(path, "a") as file:
                file.write("%s%s:\n" % (space, key))
            save_dict(item, path, indent=indent, level=level + 1)
        else:
            with open(path, "a") as file:
                file.write("%s%s: %s\n" % (space, key, str(item)))


def dict2string(dictionary, path, string="", indent=4, level=0):
    """
    :param dictionary:
    :param path:
    :param string:
    :param indent:
    :param level:
    :return:
    """
    space = " " * indent * level
    for key, item in dictionary.items():
        if isinstance(item, dict):
            string += "%s%s:\n" % (space, key)
            string = dict2string(item, path, string=string, indent=indent, level=level + 1)
        else:
            string += "%s%s: %s\n" % (space, key, str(item))
    return string


def load_yaml(yaml_path):
    """
    :param yaml_path: path to a yaml file
    :return: dictionary containing the contents of the yaml fle
    """
    yaml_dict = {}                                                              # Initialise yaml_dict
    if os.path.isfile(yaml_path):                                               # If the file exists
        with open(yaml_path) as file:                                           # Open the file
            yaml_dict = yaml.load(file)                                         # Load the file
    else:
        print("Warning: " + yaml_path + " not found.")
    return yaml_dict


def get_file_paths(directory):
    """
    :param directory:
    :return:
    """
    file_paths = []
    path = directory
    if os.path.isdir(directory):
        for item in os.listdir(directory):
            item = "/".join([path, item])
            if os.path.isdir(item):
                file_paths += get_file_paths(item)
            else:
                file_paths.append(item)
    else:
        print("Warning: " + directory + " not found.")
    return file_paths


def progress_bar(iteration, total, prefix="", suffix="", decimals=1, length=100, fill="="):
    """
    :param iteration:
    :param total:
    :param prefix:
    :param suffix:
    :param decimals:
    :param length:
    :param fill:
    :return:
    """
    try:
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / total))
        filled_length = int(length * iteration / total)
        bar = fill * filled_length + "-" * (length - filled_length)
        print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end="\r")
        if iteration == total:
            print()
    except ZeroDivisionError:
        print("None.")

def count_number_lines(filename):
    """
    Author : Alix Leroy
    :param filename: File path
    :return: Integer : Number of lines in the file
    """
    with open(filename) as f:
        for i, l in enumerate(f):
            pass
        return i + 1


def convert_number_to_letters(number):


        digits = [int(d) for d in str(number)]

        letters = []

        for digit in digits:

            if digit == 0:
                letter = "a"
            elif digit == 1:
                letter = "b"
            elif digit == 2:
                letter = "c"
            elif digit == 3:
                letter = "d"
            elif digit == 4:
                letter = "e"
            elif digit == 5:
                letter = "f"
            elif digit == 6:
                letter = "g"
            elif digit == 7:
                letter ="h"
            elif digit == 8:
                letter = "i"
            elif digit == 9:
                letter = "j"
            else:
                raise ValueError("Digit {0} not found".format(digit))

            letters.append(letter)

        return (''.join(letters))
