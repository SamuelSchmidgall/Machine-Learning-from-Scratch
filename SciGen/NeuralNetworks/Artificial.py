#!/usr/bin/env python
__author__ = "Samuel Schmidgall"
__license__ = "MIT"
__email__ = "sschmidg@masonlive.gmu.edu"

import random
import numpy as np
from SciGen.Utils.ActivationFunctions import sigmoid


class ArtificialNeuralNetwork:
    """
    Artificial Neural Network model using matrix algebra for optimization
    """
    def __init__(self, dimensions, activation=None):
        """
        Instantiate Neural Network
        :param dimensions: list(int) -> dimensions of neural network weights
        :param activation: list(function) -> list of activation functions used for forward and back propagation
        """
        if activation is None:
            activation = [sigmoid for _ in range(len(dimensions)-1)]
        if len(dimensions) <= 1:
            raise ValueError('Dimension size must be > 2')
        elif len(activation) != len(dimensions)-1:
            raise ValueError('Activation must be size: {}'.format(len(activation)-1))
        self._weights = [np.random.rand(dimensions[d+1], dimensions[d]) for d in range(len(dimensions)-1)]
        self._bias = [np.random.rand(dimensions[d+1], 1) for d in range(len(dimensions)-1)]
        self._activation = activation

    def fit(self, x, y, iterations=100, batch_ratio=0.25, learning_rate=0.01):
        """
        Train model using given training data x and y where x[i] corresponds to y[i]
        :param x: list(ndarray) -> list of input values
        :param y: list(ndarray) -> list of expected values corresponding to given input value
        :param iterations: int -> number of training iterations
        :param batch_ratio: float -> percentage of training examples to use towards training
        :param learning_rate: float -> rate at which network reacts to a given gradient
        """
        if len(x) != len(y):
            raise ValueError('Incompatible training data sizes')
        x = [np.resize(np.array(x[i]), (len(x[i]), 1)) for i in range(len(x))]
        y = [np.resize(np.array(y[i]), (len(y[i]), 1)) for i in range(len(y))]
        training_data = [(x[i], y[i]) for i in range(len(x))]
        for _ in range(iterations):
            random.shuffle(training_data)
            sample = training_data[:int(batch_ratio*len(training_data))]
            for x_sub_i, y_sub_i in sample:
                self._back_propagate(x_sub_i, y_sub_i, learning_rate)

    def predict(self, x):
        """
        Predict output for a given vector x
        :param x: ndarray -> value to predict
        :return: ndarray -> y_hat prediction
        """
        x = np.resize(np.array(x), (len(x), 1))
        return self._forward_propagate(x)[0]

    def _back_propagate(self, x, y, learning_rate):
        """
        Backpropagation algorithm using stochasitc gradient descent to update the weights and biases of network
        :param x: ndarray -> input value to backpropagate
        :param y: ndarray -> expected value used to compute initial error
        :param learning_rate: float -> rate at which network reacts to a given gradient
        """
        y_hat, a_output, z_output = self._forward_propagate(x)
        gradient = np.add(y, -1.0*y_hat)
        for _w in reversed(range(len(self._weights))):
            gradient = np.multiply(gradient, self._activation[_w](a_output[_w], derivative=True))
            self._weights[_w] = np.add(self._weights[_w], np.matmul(gradient, z_output[_w-1].T)*learning_rate)
            self._bias[_w] = np.add(self._bias[_w], gradient*learning_rate)
            gradient = np.matmul(self._weights[_w].T, gradient)

    def _forward_propagate(self, x):
        """
        Forward propagation algorithm saving hidden states for future computation
        :param x: ndarray -> value to forward propagate
        :return: ndarray -> forward propagated value
        """
        a_sub_i, z_sub_i = list(), list()
        for _w in range(len(self._weights)):
            z = np.matmul(self._weights[_w], x) + self._bias[_w]
            z_sub_i.append(z)
            x = self._activation[_w](z)
            a_sub_i.append(x)
        return x, a_sub_i, z_sub_i
