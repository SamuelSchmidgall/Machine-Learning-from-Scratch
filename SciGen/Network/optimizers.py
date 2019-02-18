import numpy as np
from copy import deepcopy


class Optimizer:
    """ Parent class for optimizer objects """
    def compute_gradients(self, layers, computations, y, y_hat):
        pass


class MSEStochasticGradientDescent(Optimizer):
    """ Stochastic Gradient Descent Optimizer using Mean Squared Error """
    def __init__(self):
        pass

    def compute_gradients(self, layers, computations, y, y_hat):
        """ Compute respective gradients at each layer """
        gradients = list()
        _error = np.multiply(np.subtract(y_hat, y), layers[-1]._activation_function(y_hat, derivative=True)) # potentially have to swap
        for _layer in reversed(range(len(layers))):
            gradients.append(layers[_layer].gradient(_error, computations[_layer]))
            if _layer != 0:
                _error = np.multiply(np.matmul(_error, layers[_layer]._weights), layers[_layer-1]._activation_function(computations[_layer]))
        return deepcopy(gradients)







