#!/usr/bin/env python
__author__ = "Samuel Schmidgall"
__license__ = "MIT"
__email__ = "sschmidg@masonlive.gmu.edu"

import random
from SciGen.NeuralNetworks.NetworkGenerator.Generator import Generator
from SciGen.NeuralNetworks.Artificial import ArtificialNeuralNetwork


class FlexANN:
    def __init__(self, d_type, io_tuples, train_data, test_data, complexity=0):
        """
        An Artificial Neural Network that generates its own parameters based on input data
        :param d_type: str -> type of data that Network will process
        :param io_tuples: list -> list of tuples corresponding to expected output given an input
        :param train_data: list -> data used for testing
        :param test_data: list -> data used for training
        :param complexity: expected complexity of model (how flexible should the network be)
        """
        self.network_data = io_tuples
        self.training_data, self.test_data = train_data, test_data
        self.dimensions = Generator(d_type, io_tuples, complexity).generator.network
        self.network = ArtificialNeuralNetwork(self.dimensions)
        self.past_parameters = list()
        self.past_parameters.append(self.dimensions)
        self._train()

    def _train(self, epochs=None):
        """
        Train neural network
        :param epochs: int -> number of training set iterations
        """
        if epochs is None:
            epochs = len(self.network_data)*1000
        for epoch in range(epochs):
            random.shuffle(self.training_data)
            for d1, d2 in self.training_data:
                self.network.back_prop(d1,d2)

    def forward_propagate(self, inputs):
        """
        Forward propagation
        :param inputs: ndarray -> LIST/NDARRAY: inputs; input values for ANN
        :return: ndarray -> forward propogated values
        """
        return self.network.forward_prop(inputs)

    def retrain(self):
        """
        If network is not performing well, give the option to retrain the network with new parameters
        :return:
        """
        pass

