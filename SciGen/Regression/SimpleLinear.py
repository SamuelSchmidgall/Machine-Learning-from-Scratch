#!/usr/bin/env python
__author__ = "Samuel Schmidgall"
__license__ = "MIT"
__email__ = "sschmidg@masonlive.gmu.edu"


class SimpleLinear:
    def __init__(self, dataset):
        """ Generate simple linear regression model """
        try:
            if self._is_valid_dataset(dataset):
                self._dataset = dataset
        except IndexError:
            raise Exception("Incompatible Dataset: REQUIRED - list of 2 variable tuples")
        self._cof = self._coefficients(self._dataset)

    def predict(self, value):
        """ Predict value based on computed regression """
        return self._cof[0] + self._cof[1]*value

    def _mean(self, values):
        """ Mean of list of data values """
        if len(values) > 0:
            return sum(values)/len(values)

    def _variance(self, values):
        """ Variance of list of data values """
        mean_v = self._mean(values)
        return sum([(x - mean_v) ** 2 for x in values])

    def _covariance(self, x, y):
        """ Covariance of list of data values """
        mean_x, mean_y, = self._mean(x), self._mean(y)
        return sum((x[val] - mean_x) * (y[val] - mean_y) for val in range(len(x)))

    def _coefficients(self, dataset):
        """ Generate polynomial coefficents for list of data values """
        x, y = [row[0] for row in dataset], [row[1] for row in dataset]
        b1 = self._covariance(x, y) / self._variance(x)
        x_mean, y_mean = self._mean(x), self._mean(y)
        b0 = y_mean - b1 * x_mean
        return [b0, b1]

    def _is_valid_dataset(self, dataset):
        """ Check if dataset contains values that can be computed """
        truth = [len(dataset[i]) == 2 and \
          (type(dataset[i][0]) == int and type(dataset[i][1]) == int) or \
          (type(dataset[i][0]) == float and type(dataset[i][1])) == float
          for i in range(len(dataset))]
        return all(truth)







