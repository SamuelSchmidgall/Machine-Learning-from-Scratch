import random
import numpy as np
from SciGen.Network.utilities import ReLU, ReLU_uniform_random


class Layer:
    """ Parent class for layer objects """
    def forward(self, x, training):
        pass

    def gradient(self, error, hidden_output):
        pass


class Linear(Layer):
    """ Linear feed-forward layer """
    def __init__(self, input_size, output_size, activation_function=ReLU, dropout_probability=0.0, weight_generator=ReLU_uniform_random):
        """ Initialization function for Linear class """
        self._activation_function = activation_function
        self._dropout_history = None # information on previous dropouts
        self._bias = weight_generator() # generate bias in same regard as weights
        self._dropout = [dropout_probability==0.0, dropout_probability] # pack it into list as dropout info
        self._weights = np.array([[weight_generator() for _ in range(input_size)] for _ in range(output_size)])

    def forward(self, x, training=False):
        """ Forward pass through linear layer """
        if not training:
            return self._activation_function(np.matmul(self._weights, self._dropout_transformation(x, training)))
        return self._activation_function(np.matmul(self._weights, self._dropout_transformation(x)))

    def gradient(self, error, hidden_output):
        """ Compute the gradient for this layer given error and hidden output """
        error = np.reshape(error, (len(error), 1))
        hidden_output = np.reshape(hidden_output, (len(hidden_output), 1))
        return np.matmul(error, hidden_output.T)

    def update_weights(self, update):
        """ Update weights """
        self._weights = np.add(self._weights, update)

    def _dropout_transformation(self, x, training=False):
        """ Perform dropout transformation provided in header data """
        if not training: # scale up if testing
            return x*(1/(1-self._dropout[1]))
        elif not self._dropout[0]:
            return x
        x_drop = list()
        for _elem in range(len(x)):
            if random.uniform(0, 1) <= self._dropout[1]:
                x_drop.append(x[_elem]*(1/(1-self._dropout[1])))
            else:
                x_drop.append(0.0)
        return np.array(x_drop)







