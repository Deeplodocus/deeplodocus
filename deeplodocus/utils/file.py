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
