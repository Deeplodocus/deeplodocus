import random

from deeplodocus.data.transformer.transformer import Transformer

class OneOf(Transformer):

    def __init__(self, augmentation_functions=None):
        Transformer.__init__(self)

        self.augmentation_functions = augmentation_functions


    def augment_image(self, image):

        random_choice = random.randint(0, len(self.augmentation_functions) -1)

        i = 0

        for key in self.augmentation_functions:
            if i == random_choice:

                image = self.__augment_image(image, key)
            i = i+ 1

        return image
