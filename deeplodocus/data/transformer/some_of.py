from .transformer import Transformer
import random

class SomeOf(Transformer):

    def __init__(self, augmentation_functions=None, random_order=False, number_augmentations = 1):
        Transformer.__init__(self)

        self.augmentation_functions = augmentation_functions
        self.random_order = random_order
        self.number_augmentations = number_augmentations


    def augment_image(self, image):

        random_choices = random.sample(range(0, len(self.augmentation_functions) -1), self.number_augmentations)

        i = 0
        for key in self.augmentation_functions:

            if i in random_choices:

                image = self.__augment_image(image, key)

        return image
