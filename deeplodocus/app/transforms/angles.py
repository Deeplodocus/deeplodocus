import math


def dangle_to_cos_and_sin(angle: float):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Convert a angle in degrees to its corresponding cos and sin values

    PARAMETERS:
    -----------

    :param angle(float): The angle in degrees to convert

    RETURN:
    -------

    :return : Cos and sin values
    """
    rad = math.radians(angle)
    return math.cos(rad), math.sin(rad)


def rangle_to_cos_and_sin(angle: float):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Convert a angle in radians to its corresponding cos and sin values

    PARAMETERS:
    -----------

    :param angle(float): The angle in radians to convert

    RETURN:
    -------

    :return : Cos and sin values
    """
    return math.cos(angle), math.sin(angle)


