from NetworkGenerator.Generator import Generator
from NeuralNetworks.ArtificialNeuralNetwork import ArtificialNeuralNetwork

class FlexANN:
    """ An artificial neural network that generates itself """
    def __init__(self, d_type, io_tuples):
        self.dimensions = Generator(d_type, io_tuples).generator.network
        self.network = ArtificialNeuralNetwork(self.dimensions)

    def _generate_network(self):
        pass

    def train(self, inputs, expected_outputs):
        pass



tups = [([0, 0], [0]),([1, 0], [0]),([0, 1], [0]),([1, 1], [1])]
fn = FlexANN('l_int', tups)
print(fn.network)



