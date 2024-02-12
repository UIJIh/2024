import timeit
from collections import OrderedDict

my_dict = OrderedDict({'kiwi': 4, 'apple': 5, 'cat': 3})

def first_method():
    layers_list = list(my_dict.values())
    layers_list.reverse()
    for layer in layers_list:
        pass

def second_method():
    for layer in reversed(my_dict.values()):
        pass

first_method_time = timeit.timeit(first_method, number=10000)
second_method_time = timeit.timeit(second_method, number=10000)

