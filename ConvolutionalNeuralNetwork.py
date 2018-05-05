import math
import numpy as np


class ConvolutionalNeuralNetwork:
    def __init__(self):
        pass


def NxN_filter(A, B):
    """ Run filter over two NxN matrices A and B """
    rowsA, colsA = A.shape
    rowsB, colsB = B.shape
    if not(rowsA == colsA == rowsB == colsB):
        raise Exception("Not nxn matrix")
    return np.add(np.ones((rowsA,rowsA)), -1*np.add(A, -1*B))

def NxMsum(A):
    """ The sum of all elements in a given NxM matrix """
    rows, cols = A.shape
    return np.matmul(np.matmul(np.array([1 for i in range(rows)]), A), np.resize(np.array([1 for i in range(cols)]), (cols, 1)))

def twoxtwo_mean_convolution(A):
    """ 2x2 convolution on matrix """
    num_rows = len(A) # number of rows im matrix A
    seg = np.array([np.array([1 if (k//2 == math.floor(i/2)) else 0 for k in range(0,num_rows, 2)]) for i in range(num_rows)]) # create segment matrix
    mat = np.matmul(A, seg) # A x segment_matrix
    convolution_mat = [[] for i in range(num_rows//2)] # empty matrix for convolution layers
    for itr in range(0,num_rows, 2): # iterate rows/2 to generate 2x2 conv mat
        t = np.array([1 if (i == itr or i == itr+1) else 0 for i in range(num_rows)]) # generate summation array
        convolution_mat[itr//2] = np.matmul(t, mat)/4 # update convolution matrix (4 because there are 4 elements in convolution)
    return np.array(convolution_mat)

