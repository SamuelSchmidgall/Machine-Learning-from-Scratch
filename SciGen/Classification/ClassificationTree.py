#!/usr/bin/env python
__author__ = "Samuel Schmidgall"
__license__ = "MIT"
__email__ = "sschmidg@masonlive.gmu.edu"
__credits__ = "Jason Brownlee"

import math


class ClassificationTree:
    """
    Classification model which consists of a series of splitting rules
    """
    def __init__(self, min_size, max_depth):
        """
        Instantiate class: ClassificaionTree
        :param min_size: int -> minimum size
        :param max_depth: int -> maximum recursion depth
        """
        self._tree = None
        self._min_size = min_size
        self._max_depth = max_depth

    def train(self, train_data):
        """
        Generate a decision tree based on given training data
        :param train_data: list -> training data
        :return:  -> decision tree
        """
        tree_root = self._optimal_split(train_data)
        self._split_node(tree_root, self._max_depth, self._min_size, 1)
        self._tree = tree_root

    def predict(self, predictor):
        """
        Predict value based on given tree and predictor (public)
        :param predictor: float -> item to be predicted
        :return: int -> predicted classification
        """
        return self._predict(self._tree, predictor)

    def _predict(self, tree, predictor):
        """
        Predict value based on given tree and predictor (private)
        :param tree: RegressionTree -> regression tree to predict using
        :param predictor: float -> item to be predicted
        :return: int -> predicted classification
        """
        if self._tree is None:
            raise Exception('Prediction cannot be made until model is trained')
        if predictor[tree['index']] < tree['value']:
            if isinstance(tree['left'], dict):
                return self._predict(tree['left'], predictor)
            else:
                return tree['left']
        else:
            if isinstance(tree['right'], dict):
                return self._predict(tree['right'], predictor)
            else:
                return tree['right']

    def _optimal_split(self, data):
        """
        Determine and generate optimal split for given set of data
        :param data: list(float) -> data to determine optimal split for
        :return: dict -> optimal split values
        """
        class_values = list(set(elem[-1] for elem in data))
        b_index, b_value, b_score, b_groups = math.inf, math.inf, math.inf, None
        for index in range(len(data[0])-1):
            for elem in data:
                groups = self._value_based_split(index, elem[index], data)
                g_ind = self._calculate_gini_index(groups, class_values)
                if g_ind < b_score:
                    b_index, b_value, b_score, b_groups = index, elem[index], g_ind, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    def _calculate_gini_index(self, groups, classes):
        """
        Calculate gini index for a given set of groups and classes
        :param groups: list(list(float)) -> list of group elements
        :param classes: list(list(float)) -> list of class elements
        :return: float -> gini index
        """
        num_instances = float(sum([len(g) for g in groups]))
        gini_index = float()
        for group in groups:
            g_length, g_sum = len(group), float()
            if g_length == 0.0:
                continue
            # scores weighted by size
            g_sum += sum([(([elem[-1] for elem in group]
                            .count(c_val))/g_length)**2 for c_val in classes])
            gini_index += (1.0 - g_sum)*(g_length/num_instances)
        return gini_index


    def _value_based_split(self, index, value, data):
        """
        Split a given set of data based on an attribute and value
        :param index: index to observe for given split
        :param value: value that split is based on
        :param data: list(float) -> data to generate value based split on
        :return: tuple(list, list) -> split left and right data
        """
        left_data, right_data = list(), list()
        for row in data:
            if row[index] < value:
                left_data.append(row)
            else:
                right_data.append(row)
        return left_data, right_data

    def _group_to_terminal_node(self, group):
        """
        Return most common output value in a given list of rows
        :param group: list -> group of elements to compute output from
        :return: float -> most comon element in group
        """
        group = [elem[-1] for elem in group]
        return max(set(group), key=group.count)

    def _split_node(self, node, max_depth, min_size, depth):
        """
        Optimally split node based on given parameters
        :param node: dict -> output from _optimal_split
        :param max_depth: int -> maximum depth
        :param min_size: -> minimum size
        :param depth: -> current depth
        """
        left, right = node['groups']
        del(node['groups'])
        if not left or not right:  # No split
            node['left'] = node['right'] = self._group_to_terminal_node(left + right)
            return
        if depth >= max_depth:
            node['left'] = self._group_to_terminal_node(left)
            node['right'] = self._group_to_terminal_node(right)
            return
        if len(left) <= min_size:
            node['left'] = self._group_to_terminal_node(left)
        else:
            node['left'] = self._optimal_split(left)
            self._split_node(node['left'], max_depth, min_size, depth+1)
        if len(right) <= min_size:
            node['right'] = self._group_to_terminal_node(right)
        else:
            node['right'] = self._optimal_split(right)
            self._split_node(node['right'], max_depth, min_size, depth+1)

