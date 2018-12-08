from deeplodocus.brain.thalamus import Thalamus
from deeplodocus.utils.flags.event import *


class Stopping(object):

    def __init__(self, params):

        print("Stopping class created")

        #Thalamus().connect(receiver=self.on_batch_end, event=DEEP_EVENT_ON_BATCH_END, expected_arguments=None)
        #Thalamus().connect(receiver=self.on_epoch_end, event=DEEP_EVENT_ON_EPOCH_END, expected_arguments=[])
        #Thalamus().connect(receiver=self.on_training_end, event=DEEP_EVENT_ON_TRAINING_END, expected_arguments=[])

    def on_batch_end(self):
        pass

    def on_epoch_end(self):
        pass

    def on_training_end(self):
        pass

