import math

class GeneratorANN:
    def __init__(self, io_tuples):
        self.network = self._generate(io_tuples)

    def _generate(self, io_tuples):
        """ Generate an ANN based on input output tuples
            LIST: io_tuples; list of tuples corresponding to expected output given an input
        """
        network = list()
        io_size = (len(io_tuples[0][0]), len(io_tuples[0][1]))
        for t in range(len(io_tuples)-1):
            if (len(io_tuples[t+1][0]), len(io_tuples[t+1][1])) != io_size:
                raise Exception("Inconsistent IO size")
        num_hidden = 1
        nsize = lambda a, ni, no, ns: int(ns/((ni+no)*a))
        if io_size[0] >= 3:
            num_hidden = int(math.ceil(math.log(io_size[0],3)))
        if (num_hidden/2)%1 == 0.5:
            # GENERATE PALINDROME ABCBA OF INCREASING SIZE IF ODD
            m_size = nsize(2, io_size[0], io_size[1], len(io_tuples))
            if m_size == 0:
                raise Exception("Insufficent data")
            network.append(m_size)
            dist = int((network[-1] - io_size[0])/(math.ceil(num_hidden/2)))
            for i in range(int(math.floor(num_hidden/2))):
                network.insert(0,m_size-dist*(i+1))
                network.insert(len(network),m_size-dist*(i+1))
        else:
            # GENERATE PALINDROME ABCCBA OF INCREASING SIZE IF EVEN
            m_size = nsize(2, io_size[0], io_size[1], len(io_tuples))
            if m_size == 0:
                raise Exception("Insufficent data")
            for i in range(2):
                network.append(m_size)
            dist = int((network[-1] - io_size[0])/(math.ceil(num_hidden/2)))
            for i in range(int(math.floor(num_hidden/2))-1):
                network.insert(0,m_size-dist*(i+1))
                network.insert(len(network),m_size-dist*(i+1))
        network.insert(0,  io_size[0])
        network.insert(len(network), io_size[1])
        return network

# todo: PRUNING
