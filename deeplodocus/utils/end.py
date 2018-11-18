class End(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Terminates the program
    """

    def __init__(self, error):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Terminates the program

        PARAMETERS:
        -----------

        :param error: Whether the program terminated with an error or not

        RETURN:
        -------

        :return: None
        """
        if error is False :
            self.__thanks_master()
        # Stop the program
        raise SystemExit(0)

    def __thanks_master(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Display thanks message

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: Universal Love <3
        """

        print("\n=================================")
        print("Thank you for using Deeplodocus !")
        print("== Made by Humans with deep <3 ==")
        print("=================================\n")
