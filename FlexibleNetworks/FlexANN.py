import random
from NetworkGenerator.Generator import Generator
from NeuralNetworks.ArtificialNeuralNetwork import ArtificialNeuralNetwork

class FlexANN:
    """ An artificial neural network that generates itself
        STRING: d_type; type of data that Network will process
        LIST: io_tuples; list of tuples corresponding to expected output given an input
        LIST: test_data; data used for testing
        LIST: train_data; data used for training
    """
    def __init__(self, d_type, io_tuples, train_data, test_data):
        self.network_data = io_tuples
        self.training_data, self.test_data = train_data, test_data
        self.dimensions = Generator(d_type, io_tuples).generator.network
        self.network = ArtificialNeuralNetwork(self.dimensions)
        self.past_parameters = list()
        self.past_parameters.append(self.dimensions)
        self._train()

    def _train(self, epochs=None):
        """ Train """
        if epochs is None:
            epochs = len(self.network_data)**3
        for epoch in range(epochs):
            random.shuffle(self.training_data)
            for d1, d2 in self.training_data:
                self.network.back_prop(d1,d2)

    def forward_propagate(self, inputs):
        return self.network.forward_prop(inputs)

    def retrain(self):
        """ If network is not performing well, give the option to retrain the network with new parameters """
        pass

