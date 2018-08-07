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
        self.inp, self.pred, self.bias = None, None, None

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
        self.inp, self.pred, self.bias = self._generate_init_values(inputs, exp_val)
        self._gradient_descent()

    def predict(self, pred_vals):
        """
        Compute multiple linear regression prediction
        :param pred_vals: ndarray -> values to predict
        :return: float -> predicted value
        """
        if self.inp is None or self.pred is None or self.bias is None:
            raise Exception('Model must be trained before prediction can occur')
        if type(pred_vals) == list:
            pred_vals = np.array(pred_vals)
        pred_vals = np.insert(pred_vals, 0, 1)
        return pred_vals.dot(self.bias)

    def _cost_function(self):
        """
        Cost function
        """
        return np.sum((self.inp.dot(self.bias) - self.pred) ** 2) / (2 * self._data_set_len)

    def _generate_init_values(self, x, y):
        """
        Generate initial values
        :param x: list(tuple(float)) -> list of x values
        :param y: list(tuple(float)) -> list of y values
        :return: ndarray -> initialized variables
        """
        return np.array([np.ones(len(x))] + [np.array([k[i] for k in x]) for i in range(self._tuple_size)]).T,\
               np.array([i[-1] for i in y]), np.zeros(self._tuple_size+1)

    def _gradient_descent(self, learning_rate=0.0001, iterations=100000):
        """
        Perform gradient decent to train model
        :param learning_rate: float -> learning rate
        :param iterations: int -> training iterations
        """
        r_itr = range(iterations)
        for _ in r_itr:
            loss = self.inp.dot(self.bias) - self.pred
            gradient = self.inp.T.dot(loss) / self._data_set_len
            self.bias = self.bias - learning_rate * gradient

    def model_efficiency(self, y_pred):
        """
        Test the efficency of trained model
        :param y_pred: ndarray -> y prediction vector
        :return: tuple(float, float) -> efficiency of given model
        """
        if type(y_pred) == list:
            y_pred = np.array(y_pred)
        mean_s_err = np.sqrt(sum((self.pred - y_pred) ** 2) / len(self.pred))
        y_mean = np.mean(self.pred)
        r2_score = 1 - (sum((self.pred - y_mean) ** 2) / sum((self.pred - y_pred) ** 2))
        return mean_s_err, r2_score

