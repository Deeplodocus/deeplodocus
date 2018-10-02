import time
from numba import jit

def test1():
    a = 0
    for i in range(15000):
        a += 1

    return a




def test2():
    a = 0
    for i in range(15000):
        a = test3(a)
    return a


def test3(a):
    a = a+1
    return a

t0 = time.time()
a = test1()
t1 = time.time()
b = test2()
t2 = time.time()


print(t1-t0)
print(t2-t1)