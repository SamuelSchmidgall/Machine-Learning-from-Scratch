#!/usr/bin/env python
__author__ = "Samuel Schmidgall"
__license__ = "MIT"
__email__ = "sschmidg@masonlive.gmu.edu"
__credits__ = "Steve Schluchter -- An amazing Linear Algebra Professor"

import cvxopt
import numpy as np


class SupportVectorClassification:
    """
    Support Vector Machine classification model
    """
    def __init__(self):
        """
        Instantiate class: SupportVectorClassification
        """
        self._bias = np.array([])
        self._weights = np.array([])

    def predict(self, predictors):
        """
        Predict output given a set of predictors
        :param predictors: ndarray -> used to calculate prediction
        :return: ndarray -> prediction
        """

    def train(self, predictors, expected_values):
        """
        Train model based on list of predictors and expected value
        :param predictors: list(ndarray) -> list of predictors to train model on
        :param expected_values: list(float) -> list of expected values for given predictors
        """
        if len(predictors) != len(expected_values):
            raise Exception('Length of predictors != length of expected values')
        self._generate_optimal_hyperplanes(predictors, expected_values)

    def _generate_optimal_hyperplanes(self, predictors, expected_values):
        """
        Find and generate optimal hyperplanes given set of predictors and expected values
        :param predictors: list(ndarray) -> list of predictors to train model on
        :param expected_values: list(float) -> list of expected values for given predictors
        """
        m = predictors.shape[0]
        k = np.array([np.dot(predictors[i], predictors[j]) for j in range(m) for i in range(m)]).reshape((m, m))
        p = cvxopt.matrix(np.outer(expected_values, expected_values)*k)
        q = cvxopt.matrix(-1*np.ones(m))
        equality_constraint1 = cvxopt.matrix(expected_values, (1,  m))
        equality_constraint2 = cvxopt.matrix(0.0)
        inequality_constraint1 = cvxopt.matrix(np.diag(-1*np.ones(m)))
        inequality_constraint2 = cvxopt.matrix(np.zeros(m))
        solution = cvxopt.solvers.qp(p, q, inequality_constraint1, inequality_constraint2,
                                     equality_constraint1, equality_constraint2)
        multipliers = np.ravel(solution['x'])
        has_positive_multiplier = multipliers > 1e-7
        sv_multipliers = multipliers[has_positive_multiplier]
        support_vectors = predictors[has_positive_multiplier]
        support_vectors_y = expected_values[has_positive_multiplier]
        if support_vectors and support_vectors_y and sv_multipliers:
            self._weights = np.sum(multipliers[i]*expected_values[i]*predictors[i] for i in range(len(expected_values)))
            self._bias = np.sum([expected_values[i] - np.dot(self._weights, predictors[i])
                                 for i in range(len(predictors))])/len(predictors)
        else:
            pass


svm = SupportVectorClassification()
y = np.array([np.array([1]), np.array([-1]), np.array([-1]), np.array([1]), np.array([-1])])
t_data = np.array([np.array([1, 1]), np.array([2, 2]), np.array([2, 3]), np.array([0, 0]), np.array([2, 4])])
svm.train(t_data, y)

