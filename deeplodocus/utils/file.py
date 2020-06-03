def get_specific_line(filename: str, index: int) -> str:
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Get a the content of a specific file's line

    PARAMETERS:
    -----------

    :param filename (str): The name of the file
    :param index (int): The index of the line (0-indexed)

    RETURN:
    -------

    :return (str): The specific line desired
    """
    with open(filename) as f:
        for i, line in enumerate(f):
            if i == index:
                return line.rstrip()            # Get the line and remove the \n at the end
            elif i > index:
                break


def compute_num_lines(filename: str) -> int:
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Get the number of lines in a file

    PARAMETERS:
    -----------

    :param filename (str): The name of the file

    RETURN:
    -------

    :return num_lines (int): Number of lines in a file
    """
    with open(filename) as f:
        for i, _ in enumerate(f):
            pass
    return i + 1
