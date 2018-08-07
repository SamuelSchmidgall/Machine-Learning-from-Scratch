#!/usr/bin/env python
__author__ = "Samuel Schmidgall"
__license__ = "MIT"
__email__ = "sschmidg@masonlive.gmu.edu"

import numpy as np


class MultipleLinear:
    def __init__(self, dataset):
        """
            Multiple Linear Regression Model
            Value that needs to be predicted should be last value of each tuple
        """
        try:
            self._tuple_size = len(dataset[0])
            if all([dataset[i] == self._tuple_size for i in range(len(dataset))]):
                self._dataset = dataset
        except IndexError:
            raise Exception("Invalid Dataser")
        self._dataset_len = len(dataset)
        self.inp, self.pred, self.bias = self._generate_init_values(dataset)
        self._gradient_descent()

    def predict(self, pred_vals):
        """ Compute multiple linear regression prediction """
        if type(pred_vals) == list:
            pred_vals = np.array(pred_vals)
        pred_vals = np.insert(pred_vals,0,1)
        return pred_vals.dot(self.bias)

    def _cost_function(self):
        """ Cost function """
        return np.sum((self.inp.dot(self.bias) - self.pred) ** 2) / (2 * self._dataset_len)

    def _generate_init_values(self, dataset):
        """ Generate initial values """
        return np.array([np.ones(len(dataset))]+ \
          [np.array([k[i] for k in dataset]) for i in range(self._tuple_size-1)]).T, \
          np.array([i[-1] for i in dataset]), np.zeros(self._tuple_size)

    def _gradient_descent(self, learning_rate=0.0001, iterations=100000):
        """ Perform gradient decent to train model """
        r_itr = range(iterations)
        for itr in r_itr:
            loss = self.inp.dot(self.bias) - self.pred
            gradient = self.inp.T.dot(loss) / self._dataset_len
            self.bias = self.bias - learning_rate * gradient

    def model_efficency(self, Y_pred):
        """ Test the efficency of trained model """
        if type(Y_pred) == list:
            Y_pred = np.array(Y_pred)
        mean_s_err = np.sqrt(sum((self.pred - Y_pred) ** 2) / len(self.pred))
        y_mean = np.mean(self.pred)
        r2_score = 1 - (sum((self.pred - y_mean) ** 2) / sum((self.pred - Y_pred) ** 2))
        return mean_s_err, r2_score

