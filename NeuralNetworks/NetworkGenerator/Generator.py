from NeuralNetworks.NetworkGenerator.GeneratorANN import GeneratorANN

data_types = {'l_float':GeneratorANN, 'l_int':GeneratorANN} #'images'

class Generator:
    def __init__(self, d_type, io_tuples, complexity):
        """ STRING: d_type; type of data that Network will process
            LIST: io_tuples; ALL of input and output value tuples
        """
        if d_type not in data_types.keys():
            raise Exception('Invalid data type')
        self.generator = data_types[d_type](io_tuples, complexity)








