import numpy as np


def tanh(value, derivative=False):
    """
    Tanh(x) function / derivative
    :param value: ndarray -> value to activate
    :param derivative: bool -> compute derivative
    :return: ndarray -> activated ndarray
    """
    if derivative:
        return 1.0 - (value ** 2.0)
    return np.tanh(value)


def sigmoid(value, derivative=False):
    """
    Sigmoid(x) function / derivative
    :param value: ndarray -> value to activate
    :param derivative: bool -> compute derivative
    :return: ndarray -> activated ndarray
    """
    if derivative:
        return value * (1.0 - value)
    return 1.0 / (1.0 + np.exp(-value))


def softplus(value, derivative=False):
    """
    Softplus(x) function / derivative
    :param value: ndarray -> value to activate
    :param derivative: bool -> compute derivative
    :return: ndarray -> activated ndarray
    """
    if derivative:
        return 1.0 / (1.0 + np.exp(-value))
    return np.log(1.0 + np.exp(value))