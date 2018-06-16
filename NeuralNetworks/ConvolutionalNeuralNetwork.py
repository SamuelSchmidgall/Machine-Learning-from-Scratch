import math
import numpy as np
from NeuralNetworks.ArtificialNeuralNetwork import ArtificialNeuralNetwork

#todo: allow for people to put in any sized images and 'fluff' them
class ConvolutionalNeuralNetwork:
    def __init__(self, dimensions):
        if len(dimensions) <= 1 or dimensions[0] <= 1:
            raise Exception("Invalid CNN dimensions")
        self.net = ArtificialNeuralNetwork(dimensions) # GENERATE <-



def check_NxN(*matrices):
    """ Makes sure all matrices are NxN and returns dimension """
    mats = list()
    for matrix in matrices:
        rowsA, colsA = matrix.shape
        if rowsA == colsA and (len(mats) == 0 or sum(mats)//len(mats)):
            mats.append(rowsA)
        else:
            raise Exception("Matrices not NxN")
    return sum(mats)//len(mats)

def NxN_filter_boxing_convolution(A, B):
    """ Filters an NxN matrix A with an MxM matrix B S.T. N,M are natural numbers and M < N """
    n = check_NxN(A)
    m = check_NxN(B)
    if n%m != 0:
        raise Exception("Matrices have incompatible dimensions {} {}".format(n, m))
    filtered_matrix = []
    for itr1 in range(n-m):
        for itr2 in range(n - m):
            filtered_matrix.append(NxMsum(NxN_filter(A[np.ix_((itr1, itr1+m),(itr2, itr2+m))], B)))
    return np.resize(np.array(filtered_matrix), (n//m, n//m))

def NxN_filter(A, B):
    """ Run filter over two NxN matrices A and B """
    n = check_NxN(A, B)
    return np.add(np.ones((n,n)), -1*np.add(A, -1*B))

def NxMsum(A):
    """ The sum of all elements in a given matrix """
    rows, cols = A.shape
    return np.matmul(np.matmul(np.array([1 for i in range(rows)]), A), np.resize(np.array([1 for i in range(cols)]), (cols, 1)))[0]

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
