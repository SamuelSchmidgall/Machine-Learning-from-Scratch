import random


def ReLU(x, derivative=False):
    """ ReLU function with corresponding derivative """
    if derivative:
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
    x[x < 0] = 0
    return x

def ReLU_uniform_random():
    """ Ideal weight starting values for ReLU """
    return random.uniform(0.005, 0.2)

def uniform_random():
    """ Generic uniform random from -n to n given output is multiplied by n """
    return random.uniform(-1, 1)








