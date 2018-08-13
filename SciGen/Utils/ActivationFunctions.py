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


def softmax(value, derivative=False):
    """
    Softmax(x) function / derivative
    :param value: ndarray -> value to activate
    :param derivative: bool -> compute derivative
    :return: ndarray -> activated ndarray
    """
    if derivative:
        s = value.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)
    return np.exp(value) / np.sum(np.exp(value))


def stable_softmax(value, derivative=False):
    """
    Softmax(x) function / derivative (softmax - except compute value in a numerically stable way)
    :param value: ndarray -> value to activate
    :param derivative: bool -> compute derivative
    :return: ndarray -> activated ndarray
    """
    if derivative:
        s = value.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)
    shift_x = value - np.max(value)
    return np.exp(shift_x)/np.sum(np.exp(shift_x))
