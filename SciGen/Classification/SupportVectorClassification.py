#!/usr/bin/env python
__author__ = "Samuel Schmidgall"
__license__ = "MIT"
__email__ = "sschmidg@masonlive.gmu.edu"
__credits__ = "Steve Schluchter -- An amazing Linear Algebra Professor"

import scipy
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

    def train(self, predictors, expected_values, iterations, learning_rate=0.01):
        """
        Train model based on list of predictors and expected value
        :param predictors: list(ndarray) -> list of predictors to train model on
        :param expected_values: list(ndarray) -> list of expected values for given predictors
        :param iterations: int -> number of training iterations
        :param learning_rate: float -> rate in which model learns
        """

    def _generate_optimal_omega_function(self, f, omega):
        """
        Generate function f(x*) such that f(x*) is continuously twice differentiable function at x*
        We want to check if gradient(f(x*)) = 0 to see if x* is a critical point by generating a gradient
         function and calculating f*(x*) in
        We also want to ensure that the Hessian of f at x* is positive definite where the Hessian
         is a matrix of second-order partial derivatives
        :param omega: list(ndarray) -> set of vectors used to calculate optimal omega function
        :return:
        """

    @staticmethod
    def _positive_semi_definite_hessian_matrix(hessian_matrix):
        """
        Determine whether or not a given hessian matrix is positive semi-definite definite
        :param hessian_matrix: ndarray -> hessian matrix to determine if positive semi-definite definite
        :return: bool -> is given matrix positive definite
        """
        eigen_values, v = scipy.linalg.eigh(hessian_matrix)
        return all([e >= 0 for e in eigen_values])

    def _maximize_hyperplane_margin(self):
        """
        Let h_sub_0 = w*x + b = -1 and h_sub_1 = w*x + b = 1
        :return:
        """

    def _hyperplane_perpendicular_distance_vector(self):
        """
        Generate a vector that is perpendicular to hyperplane(1, 2) and has
         magnitude equal to the maximum distance between h1 and h2
        :return: ndarray -> vector k in which x_sub_0 + k = 1
        """
        return self._normalized_vector(self._weights)*(2/self._calculate_vector_norm(self._weights))

    def _hyperplane_constraint(self, predictor, expected_classification):
        """
        Determine if given predictor satisfies the hyperplane constraint
        y_sub_i * (weights * x_sub_i + bias) >= 1 for all 1 <= i <= n
        :param predictor: ndarray -> vector to observe
        :param expected_classification: float -> expected classification (1, -1)
        :return: bool -> constraint satisfied
        """
        return expected_classification*(self._weights*predictor + self._bias) >= 1

    def _hyperplane_margin(self, vector1, vector2):
        """
        Calculate margin between two vectors -- vectors distance from hyperplane
        :param vector1: ndarray -> vector that is being observed
        :param vector2: ndarray -> hyperplane vector representation
        :return: float -> computed margin
        """
        return 2*self._calculate_vector_norm(self._calculate_projection(vector1, vector2))

    def _calculate_projection(self, vector1, vector2):
        """
        Calculate the projection of vector1 onto vector2
        :param vector1: ndarray -> projection of vector1 onto vector2
        :param vector2: ndarray -> vector getting projected onto
        :return: ndarray -> projection vector
        """
        vector2_norm = self._normalized_vector(vector2)
        return np.dot(vector2_norm, vector1)*vector2_norm

    def _calculate_vector_norm(self, vector):
        """
        Length of a given vector
        :param vector: ndarray -> vector to calculate norm of
        :return: float -> norm of vector
        """
        return np.sqrt(np.matmul(vector, vector.T))

    def _normalized_vector(self, vector):
        """
        Generate a normalized vector
        :param vector: ndarray -> vector used to generate norm
        :return: ndarray -> normalized vector
        """
        return vector/self._calculate_vector_norm(vector)

    @staticmethod
    def _hessian_matrix(vector):
        """
        Calculate the hessian matrix with finite differences
        https://stackoverflow.com/questions/31206443/numpy-second-derivative-of-a-ndimensional-array
        :param vector: ndarray -> ndarray to calculate hessian matrix of
        :return: ndarray -> shape=(x.dim, x.ndim)+x.shape where
         the array[i, j, ...] corresponds to the second derivative x_ij
        """
        x_grad = np.gradient(vector)
        hessian = np.empty((vector.ndim, vector.ndim) + vector.shape, dtype=vector.dtype)
        for k, grad_k in enumerate(x_grad):
            tmp_grad = np.gradient(grad_k)
            for l, grad_kl in enumerate(tmp_grad):
                hessian[k, l, :, :] = grad_kl
        return hessian


svm = SupportVectorClassification()
print(svm._hyperplane_margin(np.array([-1, -1]), np.array([2, 1])))

