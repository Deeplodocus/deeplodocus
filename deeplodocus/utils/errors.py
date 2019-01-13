from deeplodocus.utils.notification import Notification


def error_entry_array_size(d: dict, error_type):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Print the details of the data dictionary when raising an issue of array size

    PARAMETERS:
    -----------

    :param d (dict): The data dictionary
    :param error_type: The error type

    RETURN:
    -------

    :return: None
    """

    text = "All your entries do not have the same number of instances : \n"

    if "inputs" in d.keys():
        text += "Inputs : " + str(len(d["inputs"])) + "\n"

    if "labels" in d.keys():
        text += "Labels : " + str(len(d["labels"])) + "\n"

    if "additional_data" in d.keys():
        text += "Additional data : " + str(len(d["additional_data"])) + "\n"

    text += "Error type : " + str(error_type)

    Notification(DEEP_NOTIF_FATAL, text)
