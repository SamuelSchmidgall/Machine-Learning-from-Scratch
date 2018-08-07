#!/usr/bin/env python
__author__ = "Samuel Schmidgall"
__license__ = "MIT"
__email__ = "sschmidg@masonlive.gmu.edu"

import numpy as np


class MultipleLinear:
    def __init__(self):
        """
        Multiple Linear Regression Model
        """
        self._tuple_size = None
        self._data_set_len = None
        self._inp, self._pred, self._bias = None, None, None

    def train(self, inputs, exp_val):
        """
        Train the given model based on inputs and expected value
        :param inputs: input value which is expected to return corresponding exp_val
        :param exp_val: expected value for a given input
        """
        if len(inputs) != len(exp_val) <= 0:
            raise Exception('Invalid Data Set')
        self._tuple_size = len(inputs[0])
        self._data_set_len = len(inputs)
        self._inp, self._pred, self._bias = self._generate_init_values(inputs, exp_val)
        self._gradient_descent()

    def predict(self, predicted_values):
        """
        Compute multiple linear regression prediction
        :param predicted_values: ndarray -> values to predict
        :return: float -> predicted value
        """
        if self._inp is None or self._pred is None or self._bias is None:
            raise Exception('Model must be trained before prediction can occur')
        if type(predicted_values) == list:
            predicted_values = np.array(predicted_values)
            predicted_values = np.insert(predicted_values, 0, 1)
        return predicted_values.dot(self._bias)

    def _cost_function(self):
        """
        Cost function for regression
        """
        return np.sum((self._inp.dot(self._bias) - self._pred) ** 2) / (2 * self._data_set_len)

    def _generate_init_values(self, x, y):
        """
        Generate initial values
        :param x: list(tuple(float)) -> list of x values
        :param y: list(float) -> list of y values
        :return: ndarray -> initialized variables
        """
        return np.array([np.ones(len(x))] + [np.array([k[i] for k in x]) for i in range(self._tuple_size)]).T, \
               np.array(y), np.zeros(self._tuple_size+1)

    def _gradient_descent(self, learning_rate=0.0001, iterations=100000):
        """
        Perform gradient decent to train model
        :param learning_rate: float -> learning rate
        :param iterations: int -> training iterations
        """
        r_itr = range(iterations)
        for _ in r_itr:
            loss = self._inp.dot(self._bias) - self._pred
            gradient = self._inp.T.dot(loss) / self._data_set_len
            self._bias = self._bias - learning_rate * gradient

    def model_efficiency(self, y_prediction):
        """
        Test the efficency of trained model
        :param y_pred: ndarray -> y prediction vector
        :return: tuple(float, float) -> efficiency of given model
        """
        if type(y_prediction) == list:
            y_prediction = np.array(y_prediction)
        mean_s_err = np.sqrt(sum((self._pred - y_prediction) ** 2) / len(self._pred))
        y_mean = np.mean(self._pred)
        r2_score = 1 - (sum((self._pred - y_mean) ** 2) / sum((self._pred - y_prediction) ** 2))
        return mean_s_err, r2_score

