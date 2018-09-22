#Importing JIT cab be useful to reduce the processing time
from numba import jit

#Use decorator to activate JIT
@jit
def example_function(data, parameters):

    transformed_data = data -1

    return transformed_data