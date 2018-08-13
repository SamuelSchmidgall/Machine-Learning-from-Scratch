#!/usr/bin/env python
__author__ = "Samuel Schmidgall"
__license__ = "MIT"
__email__ = "sschmidg@masonlive.gmu.edu"
__credits__ = "Nick Becker"

import numpy as np
from SciGen.Utils.ActivationFunctions import sigmoid


class LogisticRegression:
    """
    Logistic Regression model
    """
    def __init__(self):
        """
        Instantiate class Logistic regression
        """
        self._weights = None

    def predict(self, inputs):
        """
        Inputs that will be used to predict output
        :param inputs: ndarray -> values to be used for prediction
        :return: float -> value between 0 and 1 to indicate prediction from model
        """
        if self._weights is None:
            raise Exception('Model must be trained before prediction can occur')
        _values = np.dot(inputs, self._weights)
        _prediction = sigmoid(_values)
        return _prediction

    def train(self, inputs, exp_val, steps=100000, learning_rate=0.01, add_intercept=False):
        """
        Train logistic model based on a set of inputs and expected values
        :param inputs: ndarray -> array of inputs
        :param exp_val: ndarray -> array of expected values for given inputs
        :param steps: int -> number of steps that training model should iterate on
        :param learning_rate: float -> rate at which model should learn
        :param add_intercept: bool -> should intercept be added
        """
        if add_intercept:
            intercept = np.ones((inputs.shape[0], 1))
            inputs = np.hstack((intercept, inputs))
        _weights = np.zeros(inputs.shape[1])
        for step in range(steps):
            _val = np.dot(inputs, _weights)
            _grad = np.dot(inputs.T, exp_val - sigmoid(_val))
            _weights += learning_rate*_grad
        self._weights = _weights

    @staticmethod
    def _log_likelihood(inp, exp_val, weights):
        """
        Calculates Log Likelihood function based on inputs, expected value and weights
        :param inp: ndarray -> input vector that has exp_val has expected value
        :param exp_val: ndarray -> expected value from input
        :param weights: ndarray -> weight matrix
        :return: ndarray -> log likelihood calculated value
        """
        _values = np.dot(inp, weights)
        return np.sum(exp_val*_values - np.log(1 + np.exp(_values)))


