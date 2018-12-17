import numpy as np
from numpy import cov
from numpy.linalg import eig

class PCA:
    """ Principal Component Analysis """
    def __init__(self):
        pass

    def predict(self, feature_matrix):
        column_means = list()
        for column in np.array(feature_matrix).T:
            column_means.append(sum(column)/len(column))
        for _i in range(len(feature_matrix)):
            for _j in range(len(feature_matrix[_i])):
                feature_matrix[_i][_j] -= column_means[_j]
        covariance = cov(np.array(feature_matrix).T)
        values, vectors = eig(covariance)
        return vectors.T.dot(np.array(feature_matrix).T)















