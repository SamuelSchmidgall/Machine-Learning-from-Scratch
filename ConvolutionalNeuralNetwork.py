import math
import numpy as np


def twoxtwoConvolution(A):
    """ Lightning fast 2x2 convolution on matrix """
    l_arr = len(A)
    seg = np.array([np.array([1 if (k//2 == math.floor(i/2)) else 0 for k in range(0,l_arr, 2)]) for i in range(l_arr)])
    mat = np.matmul(A, seg)
    convolution_mat = [[] for i in range(l_arr//2)]
    for itr in range(0,l_arr, 2):
        t = np.array([1 if (i == itr or i == itr+1) else 0 for i in range(l_arr)])
        convolution_mat[itr//2] = (np.matmul(t, mat))
    return np.array(convolution_mat)


B = np.array([np.array([1,1,2,2,3,3,4,4]),np.array([1,1,2,2,3,3,4,4]),np.array([5,5,6,6,7,7,8,8]),np.array([5,5,6,6,7,7,8,8]),
              np.array([9,9,10,10,11,11,12,12]),np.array([9,9,10,10,11,11,12,12]),np.array([13,13,14,14,15,15,16,16]),np.array([13,13,14,14,15,15,16,16])])

twoxtwoConvolution(B)

