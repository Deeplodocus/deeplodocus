import random

#
# RANDOM FUNCTION EXAMPLE
#
def random_example_function(data, param_min, param_max):
    parameters = random.uniform(param_min, param_max)
    transformed_data = example_function(data, parameters)
    transform = ["example_function", example_function, {"parameters": parameters}]
    return transformed_data, transform


#
# FUNCTION EXAMPLE
#
def example_function(data, parameters):
    return data, None

