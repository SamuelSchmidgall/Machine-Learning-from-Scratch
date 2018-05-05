import math
import numpy as np


def twoxtwo_convolution(A):
    """ 2x2 convolution on matrix """
    num_rows = len(A) # number of rows im matrix A
    seg = np.array([np.array([1 if (k//2 == math.floor(i/2)) else 0 for k in range(0,num_rows, 2)]) for i in range(num_rows)]) # create segment matrix
    mat = np.matmul(A, seg) # A x segment_matrix
    convolution_mat = [[] for i in range(num_rows//2)] # empty matrix for convolution layers
    for itr in range(0,num_rows, 2): # iterate rows/2 to generate 2x2 conv mat
        t = np.array([1 if (i == itr or i == itr+1) else 0 for i in range(num_rows)]) # generate summation array
        convolution_mat[itr//2] = (np.matmul(t, mat)) # update convolution matrix
    return np.array(convolution_mat)

