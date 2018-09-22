import time

def __is_float(data):

    if "." in data :
        try:
            float(data)
            return True
        except ValueError:
            return False

    else:
        return False


def __is_integer( data):


    try:
        int(data)
        return True
    except ValueError:
        return False



def get_int_or_float(v):
    try:
        number_as_float = float(v)
        number_as_int = int(number_as_float)
        return "integer" if number_as_float == number_as_int else "float"
    except ValueError:
        return False


a = "1."

t0 = time.time()
print(__is_float(a))
print(__is_integer(a))

t1 = time.time()

print(get_int_or_float(a))
t2 = time.time()

print(t1-t0)
print(t2-t1)