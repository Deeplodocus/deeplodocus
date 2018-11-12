from deeplodocus.utils.flags import *

class OverWatchMetric(object):


    def __init__(self, name:str=TOTAL_LOSS, condition:int = DEEP_COMPARE_SMALLER):
        self.name = name
        self.value = 0.0
        self.condition = condition

    def set_value(self, value:float):
        self.value = value

    def get_name(self):
        return self.name

    def get_value(self):
        return self.value

    def get_condition(self):
        return self.condition