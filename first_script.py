# def updated(dictionary, **kwargs):
#
#     new = dictionary.copy()
#     new.update(kwargs)
#     return new
#
# def updated_my(req, **kwargs):
#     new_dict = {}
#     new_dict.update(req)
#     for key, value in kwargs.items():
#         new_dict[key] = value
#     return new_dict
#
# old_dict = {'a': 1, 'b': [False]}
# new_dict_t = updated(old_dict, a=2, b=[True], c=None)
# new_dict_my = updated(old_dict, a=2, b=[True], c=None)
#
#
# def double(function):
#     def inner(argument):
#         return function(function(argument))
#     return inner
#
# def multiply_by_five(x):
#     return x * 5
#
# users = ["Vader_Darth", "Luke_Skywalker", "Boba_Fett"]
# def get_first_name(name: str):
#     return name.split("_")[0]
# def sort_by(func, ls):
#     print(ls)
#     return sorted(ls, key=func)
#
# import math
# # BEGIN (write your solution here)
# def is_prime(num):
#     if num < 4:
#         return True
#     is_num_prime = True
#     counter = 1
#     ceiling = int(math.sqrt(num))
#     print(f"ceiling = {ceiling}")
#     while (is_num_prime) and (counter < ceiling):
#         print(f"counter before adding = {counter}")
#         counter += 1
#         print(f"counter after adding = {counter}")
#         if num % counter == 0:
#             print("TOGGLE")
#             is_num_prime = False
#     return is_num_prime
#
# def say_prime_or_not(num):
#     if is_prime(num):
#         print("yes")
#     else:
#         print("no")
#     pass
#
# numbers = [2, 3, 4, 5, 6, 7]
# #print(list(map(say_prime_or_not, numbers)))
# from functools import reduce
# # print(sum([abs(-6), abs(3)]))
# import operator
# # print(reduce(lambda x, y: sum([abs(x), abs(y)]), numbers))
#
# city = {
#     'Pine': {
#         '5': 'School #42',
#     },
#     'Elm': {
#         '13': {
#             '1': 'Appartments #2, Elm st.13',
#         },
#     },
# }
# def walk(dictionary, keys):
#     return map(operator.getitem, dictionary, keys)
#
#
# def partial_apply(func, name):
#     def inner(surname):
#         print("AAInnerFuncBeforeCall")
#         func(name, surname)
#         print("AAInnerFuncAfterCall")
#     return inner
# def add(x, y):
#     return x+y



def memoized(func):
    memo = {}
    def wrapper(x):
        if x not in memo:
            result = func(x)
            memo[x] = result
            return result
        return memo[x]
    return wrapper
@memoized
def f(x):
    print('Calculating...')
    return x * 10

counter = [0]
@memoized
def xor(byte):
    counter[0] += 1
    return 255 ^ byte

# assert xor(xor(42)) == 42
# print(counter)
# assert counter == [2]
# print(counter)
# print(xor(42))
# print(xor(xor(42)))
# assert xor(42) + xor(xor(42)) == 255
# print(counter)
# assert counter == [2]

from functools import wraps
def memoizing_2(predicate):
    def wrapper(function):
        memory = []
        results = []
        @wraps(function)
        def inner(arg):
            nonlocal memory
            nonlocal results
            nonlocal predicate
            if arg not in memory:
                memory.append(arg)
                results.append(function(arg))
                if len(memory) > predicate:
                    memory.pop(0)
                    results.pop(0)
                return results[memory.index(arg)]
            elif arg in memory:
                return results[memory.index(arg)]
        return inner
    return wrapper

def memoizing(limit):
    """
    Make decorator that will remember recent results of function (up to limit).
    Arguments:
        limit, maximum number of results to remember
    Returns:
        memoizing decorator
    """
    def wrapper(function):
        """
        Memoize function.
        Arguments:
            function, it will be memoized
        Returns:
            memoized version of function
        """
        memory = {}
        order = []

        @wraps(function)
        def inner(argument):
            memoized_result = memory.get(argument)
            if memoized_result is None:
                memoized_result = function(argument)
                memory[argument] = memoized_result
                order.append(argument)
                if len(order) > limit:
                    oldest_argument = order.pop(0)
                    memory.pop(oldest_argument)
            return memoized_result
        return inner
    return wrapper

def reverse_range(begin, end):
    return list(range(end, begin-1, -1))
# print(reverse_range(1 ,1))


def length(ls: list) -> int:
    if not ls:
        return 0
    head, *tail = ls
    if not tail: # Срабатывает когда tail пустой
        return 1
    return 1 + length(tail)

def reverse_range(begin, end):
    if end - begin == 0:
        return [end]
    return [end] + reverse_range(begin, end-1)

def filter_positive(ls: list) -> list:
    if not ls:
        return []
    head, *tail = ls
    if not tail: # Срабатывает когда tail пустой
        return [] if head < 0 else [head]
    return ([] if head < 0 else [head]) + filter_positive(tail)
def filter_positive(list):
    if not list:
        return []
    head, *tail = list
    if head > 0:
        return [head] + filter_positive(tail)
    return filter_positive(tail)

def odds_from_odds(ls) -> list:
    return [[i for i in inner_list[::2]] for inner_list in ls[::2]]

def keep_odds_from_odds(ls) -> list:
    if not ls:
        return None
    ls[:] = [i for i in ls[::2]]
    for inner_list_index in range(len(ls)):
        ls[inner_list_index][:] = [i for i in ls[inner_list_index][::2]]

def odds(source):
    is_odd_position = lambda pair: pair[0] % 2 == 0  # noqa: E731
    get_value = lambda pair: pair[1]  # noqa: E731
    return list(map(
        get_value,
        (filter(is_odd_position, enumerate(source)))
    ))


def odds_from_odds(list_of_lists):
    return list(map(odds, odds(list_of_lists)))

# Альтернативное решение с помощью itemgetter
# https://docs.python.org/3/library/operator.html#operator.itemgetter
#
# odds = itemgetter(slice(None, None, 2))
#
# def odds_from_odds(list_of_lists):
#     return list(map(odds, odds(list_of_lists)))
def keep_odd(some_list):
    index = 0
    for i in range(len(some_list)):
        if i % 2 == 1:
            some_list.pop(index)
        else:
            index += 1
def keep_odds_from_odds(list_of_lists):
    keep_odd(list_of_lists)
    for one_list in list_of_lists:
        keep_odd(one_list)

# xs = [1, 3, 5]
# ys = list(range(6, 20))
# ff = [7, 9, 11, 13, 15, 17, 19]
# print(xs)
# t = list(map(xs.append, filter(lambda y: y % 2 == 1, ys)))
# print(f"t {t}")
# print(xs)
# print(all([]))

def is_int(x):
    return isinstance(x, int)
def each2d(test, matrix):
    return all((all((test(i) for i in outer)) for outer in matrix))
def some2d(test, matrix):
    return any((any((test(i) for i in outer)) for outer in matrix))
def sum2d(test, matrix):
    return sum((sum((i for i in outer if test(i))) for outer in matrix))

def each2d(test, matrix):
    return all(
        test(cell)
        for row in matrix
        for cell in row
    )


def some2d(test, matrix):
    return any(
        test(cell)
        for row in matrix
        for cell in row
    )


def sum2d(test, matrix):
    return sum(
        cell
        for row in matrix
        for cell in row if test(cell)
    )
data = [[0, 1, 3], [1, 9]] #[[1, 8], [0, 'da']]
print("each")
fff = sum2d(is_int, data)

print(fff)
