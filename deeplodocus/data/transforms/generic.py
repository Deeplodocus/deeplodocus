def scale(item, multiply=1, divide=1):
    return item * multiply / divide, None


def bias(item, plus=0, minus=0):
    return item + plus - minus, None
