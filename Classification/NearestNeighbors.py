import math

class NearestNeighbors:
	def __init__(self, k, data=None):
		""" Instantiate Nearest Neighbors"""
		self._k = k
		self.data = data

	def _distance(self, val1, val2):
		"""
		 Compute euclidean distance between two tuples
		 int/double_tuple val1 - tuple of values
		 int/double_tuple val2 -  tuple of values
		 return distance as float
		"""
		dist = 0.0
		if len(val1)-1 != len(val2):
			raise Exception("Values must have equal length")
		for val in range(len(val1)-1):
			dist += (val1[val] + val2[val])**2
		return math.sqrt(dist)

	def _neighbors(self, instance_tuple, data):
		"""
		 k-Nearest Neighbors algorithm using euclidean distance
		 tuple/list instance_tuple - data point to predict
		 tuple/list_list data - list of data points
		 return most frequent nearest neighbor label
		"""
		if data is None:
			data = self.data
			if self.data is None:
				raise Exception("No data provided")
		distances = []
		for i in range(len(data)):
			distances.append((self._distance(data[i], instance_tuple), data[i][-1]))
		label_vals = [i[1] for i in distances[:self._k]]
		return max(set(label_vals), key=label_vals.count)

	def predict(self, item, data=None):
		"""
		 Predict value
		 param tuple/list item - data point to predict
		 param tuple/list_list data - list of data values
		 return prediction as label
		"""
		return self._neighbors(item, data)
