#!/usr/bin/env python
__author__ = "Samuel Schmidgall"
__license__ = "MIT"
__email__ = "sschmidg@masonlive.gmu.edu"


class SimpleLinear:
    def __init__(self):
        """
        Generate simple linear regression model
        """
        self._cof = None
        self._data_set = None

    def train(self, inputs, exp_val):
        """
        Generate linear regression coefficients based on given data set
        :param inputs: list(float) -> list of input x values
        :param exp_val: list(float) -> list of output y values that are expected
        :return:
        """
        self._cof = self._coefficients(inputs, exp_val)

    def predict(self, value):
        """
        Predict value based on computed regression
        :param value: float -> value to predict
        :return: predicted value from regression model
        """
        return self._cof[0] + self._cof[1]*value

    @staticmethod
    def _mean(values):
        """
        Mean of list of data values
        :param values: list(float) -> values to generate mean from
        :return: float -> mean of values
        """
        if len(values) > 0:
            return sum(values)/len(values)

    def _variance(self, values):
        """
        Variance of list of data values
        :param values: list(float) -> list of values to calculate variance from
        :return: float -> variance of values
        """
        mean_v = self._mean(values)
        return sum([(x - mean_v) ** 2 for x in values])

    def _covariance(self, x, y):
        """
        Covariance of list of data values
        :param x: list(float) -> list of x variables
        :param y: list(float) -> list of y variables
        :return: float -> covariance of lists
        """
        mean_x, mean_y, = self._mean(x), self._mean(y)
        return sum((x[val] - mean_x) * (y[val] - mean_y) for val in range(len(x)))

    def _coefficients(self, x, y):
        """
        Generate polynomial coefficients for list of data values
        :param data_set: list(tuple(x, y)) -> variables to train model on
        :return: list(float) -> list of coefficients
        """
        b1 = self._covariance(x, y) / self._variance(x)
        x_mean, y_mean = self._mean(x), self._mean(y)
        b0 = y_mean - b1 * x_mean
        return [b0, b1]
