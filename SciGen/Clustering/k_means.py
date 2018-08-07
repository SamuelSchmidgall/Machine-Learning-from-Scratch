import math
import random
import numpy as np


class KMeans:
    def __init__(self, num_categories, tolerance=0.0001, max_iterations=500):
        self._classes = dict()
        self._centroids = dict()
        self._k = num_categories
        self._tolerance = tolerance
        self._max_iterations = max_iterations

    def predict(self, inputs):
        """
        Predict class for given inputs
        :param inputs: list(float) -> input values for prediction
        :return: int -> classification
        """
        inputs = [np.array(inp) for inp in inputs]
        distances = [np.linalg.norm(inputs - self._centroids[centroid]) for centroid in self._centroids]
        classification = distances.index(min(distances))
        return classification

    def cluster(self, inputs):
        """
        Generate unsupervised classification clusters based on K-Means using euclidean distance
        :param inputs: list(float) -> input values for clustering
        """
        if len(inputs) < self._k:
            raise Exception('length of x must be larger than num_categories for method: KMeans.cluster')
        self._instantiate_centroids(inputs)
        for _ in range(self._max_iterations):
            self._classes = dict()
            for k in range(self._k):
                self._classes[k] = list()
            self._update_classes(inputs)
            for classification in self._classes:
                self._centroids[classification] = np.average(self._classes[classification], axis=0)
            if self._is_optimal():
                break

    def _is_optimal(self):
        """
        Determine if the given centroids are optimal
        :return: bool -> if centroids are optimal
        """
        is_optimal = True
        for centroid in self._centroids:
            original_centroid = dict(self._centroids)[centroid]
            curr = self._centroids[centroid]
            if np.sum((curr - original_centroid) / original_centroid * 100.0) > self._tolerance:
                is_optimal = False
        return is_optimal

    def _update_classes(self, inputs):
        """
        Update classes based on given inputs
        :param inputs: list(float) -> list of inputs to update classes on
        """
        for item in inputs:
            distances = [np.linalg.norm(item - self._centroids[centroid]) for centroid in self._centroids]
            classification = distances.index(min(distances))
            self._classes[classification].append(item)

    def _instantiate_centroids(self, inputs):
        """
        Instantiate centroids from random samples of inputs
        :param inputs: list(float) -> list of input values
        """
        inputs = [np.array(inp) for inp in inputs]
        random.shuffle(inputs)
        for cluster in range(self._k):
            self._centroids[cluster] = inputs[cluster]

    @staticmethod
    def _euclidean_distance(vector_1, vector_2):
        """
        Determine the euclidean distance between two vectors represented as lists
        :param vector_1: list(float) -> first vector in our equation
        :param vector_2: list(float) -> second vector in our equation
        :return: float -> euclidean distance
        """
        if len(vector_1) != len(vector_2):
            raise Exception('vectors must be same size for method: KMeans._euclidean_distance')
        return math.sqrt(sum((vector_1[i] + vector_2[i])**2 for i in range(len(vector_1))))
