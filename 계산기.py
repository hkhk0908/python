def sum(a, b):
    return a + b


def sub(a, b):
    return a - b


def div(a, b):
    return a / b


def mod(a, b):
    return a % b


def mul(a, b):
    return a * b


a, b = input("a,b:").split()
a = int(a)
b = int(b)
print(a, '+', b, '=', sum(a, b))
print(a, '-', b, '=', sub(a, b))
print(a, '/', b, '=', div(a, b))
print(a, '%', b, '=', mod(a, b))
print(a, '*', b, '=', mul(a, b))
