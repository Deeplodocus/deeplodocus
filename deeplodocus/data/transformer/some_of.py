from .transformer import Transformer
import random

class SomeOf(Transformer):

    def __init__(self, config):
        Transformer.__init__(self, config)

        if hasattr(config, "number_transformations_min"):
            self.number_transformation_min = config.number_transformations_min
        else:
            self.number_transformation_min = 1

        if hasattr(config, "number_transformations_max"):
            self.number_transformation_max = config.number_transformations_max
        else:
            self.number_transformation_max = len(self.list_transforms)

        if hasattr(config, "number_transformations"):
            self.number_transformation = config.number_transformations
        else:
            self.number_transformation = None

    def transform(self, transformed_data, index, data_type):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Transform the data using the Some Of transformer

        PARAMETERS:
        -----------

        :param transformed_data: The data to transform
        :param index: The index of the data
        :param data_type: The data_type

        RETURN:
        -------

        :return transformed_data: The transformed data
        """
        transforms = []

        if self.last_index == index:
            transforms = self.last_transforms

        else:
            if self.number_transformation is not None:
                number_transforms_applied = self.number_transformation
            else:
                number_transforms_applied = random.randint(self.number_transformation_min, self.number_transformation_max)

            index_transforms_applied = random.sample(range(len(self.list_transforms)), number_transforms_applied ).sort()       # Sort the list numerically

            for index in index_transforms_applied:
                transforms.append(self.list_transforms[index])

        # Reinitialize the last transforms
        self.last_transforms = []


        # Apply the transforms
        for transform in transforms:

            transform_name = transform[0]
            transform_method = transform[1]  # Create a generic alias for the transform method
            transform_args = transform[2]  # Dictionary of arguments
            transformed_data, last_method_used = transform_method(transformed_data, **transform_args)       # Apply the transform

            # Update the last transforms used and the last index
            if last_method_used is None:
                self.last_transforms.append([transform_name, transform_method, transform_args])

            else:
                self.last_transforms.append(last_method_used)

        # Update the last index
        self.last_index = index


        return transformed_data



