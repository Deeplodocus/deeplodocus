class TransformData(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Store the parameters of a transform.
    Stores :
        - name
        - method
        - module_path
        - kwargs

    /!\ This is a read-only class
    """

    def __init__(self, name: str, method: callable, module_path: str, kwargs: dict):

        self.__name = name
        self.__method = method
        self.__module_path = module_path
        self.__kwargs = kwargs

    @property
    def name(self) -> str:
        return self.__name

    @property
    def method(self) -> callable:
        return self.__method

    @property
    def module_path(self) -> str:
        return self.__module_path

    @property
    def kwargs(self) -> dict:
        return self.__kwargs
