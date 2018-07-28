#!/usr/bin/env python
__author__ = "Samuel Schmidgall"
__license__ = "MIT"
__email__ = "sschmidg@masonlive.gmu.edu"

from SciGen.NeuralNetworks.NetworkGenerator.GeneratorANN import GeneratorANN

data_types = {'l_float': GeneratorANN, 'l_int': GeneratorANN}  # 'images'


class Generator:
    def __init__(self, d_type, io_tuples, complexity):
        """
        Instantate Generator class
        :param d_type: str -> type of data that Network will process
        :param io_tuples: list -> ALL of input and output value tuples
        :param complexity: expected complexity of model (how flexible should the network be)
        """
        if d_type not in data_types.keys():
            raise Exception('Invalid data type')
        self.generator = data_types[d_type](io_tuples, complexity)

    def valid_data_types(self):
        """
        Generate list of valid data types
        :return: list(str) -> list of valid data types
        """
        return data_types.keys()





