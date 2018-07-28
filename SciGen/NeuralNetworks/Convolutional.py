#!/usr/bin/env python
__author__ = "Samuel Schmidgall"
__license__ = "MIT"
__email__ = "sschmidg@masonlive.gmu.edu"

import math
import numpy as np
from SciGen.NeuralNetworks.Artificial import ArtificialNeuralNetwork


class ConvolutionalNeuralNetwork:
    """
    ConvolutionalNeuralNetwork; Extension of ArtificialNeuralNetwork class
    """
    def __init__(self, dimensions):
        """
        Instantiate ConvolutionalNeuralNetwork
        :param dimensions: list(int) -> dimensions of neural network weights
        """
        if len(dimensions) <= 1 or dimensions[0] <= 1 or len(dimensions[0]) != 2 \
                or math.log(dimensions[0][0], 2) % 1 != 0.0 or math.log(dimensions[0][1], 2) % 1 != 0.0:
            raise Exception("Invalid CNN dimensions")
        self.net = ArtificialNeuralNetwork(dimensions) # GENERATE <-


def check_nxn(*matrices):
    """ Makes sure all matrices are NxN and returns dimension """
    mats = list()
    for matrix in matrices:
        rows_a, cols_a = matrix.shape
        if rows_a == cols_a and (len(mats) == 0 or sum(mats)//len(mats)):
            mats.append(rows_a)
        else:
            raise Exception("Matrices not NxN")
    return sum(mats)//len(mats)


def nxn_filter_boxing_convolution(a, b):
    """ Filters an NxN matrix A with an MxM matrix bias S.T. N,M are natural numbers and M < N """
    n = check_nxn(a)
    m = check_nxn(b)
    if n % m != 0:
        raise Exception("Matrices have incompatible dimensions {} {}".format(n, m))
    filtered_matrix = []
    for itr1 in range(n-m):
        for itr2 in range(n - m):
            filtered_matrix.append(nxm_sum(nxn_filter(a[np.ix_((itr1, itr1+m), (itr2, itr2+m))], b)))
    return np.resize(np.array(filtered_matrix), (n//m, n//m))


def nxn_filter(a, b):
    """ Run filter over two NxN matrices A and bias """
    n = check_nxn(a, b)
    return np.add(np.ones((n, n)), -1*np.add(a, -1*b))


def nxm_sum(a):
    """ The sum of all elements in a given matrix """
    rows, cols = a.shape
    return np.matmul(np.matmul(np.array([1 for _ in range(rows)]), a),
                     np.resize(np.array([1 for _ in range(cols)]), (cols, 1)))[0]


def twoxtwo_mean_convolution(A):
    """ 2x2 convolution on matrix """
    num_rows = len(A) # number of rows im matrix A
    seg = np.array([np.array([1 if (k//2 == math.floor(i/2))
                              else 0 for k in range(0, num_rows, 2)]) for i in range(num_rows)])
    mat = np.matmul(A, seg) # A x segment_matrix
    convolution_mat = [[] for i in range(num_rows//2)] # empty matrix for convolution layers
    for itr in range(0,num_rows, 2):  # iterate rows/2 to generate 2x2 conv mat
        t = np.array([1 if (i == itr or i == itr+1) else 0 for i in range(num_rows)])  # generate summation array
        # update convolution matrix (4 because there are 4 elements in convolution)
        convolution_mat[itr//2] = np.matmul(t, mat)/4
    return np.array(convolution_mat)
