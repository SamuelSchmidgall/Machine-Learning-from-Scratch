import unittest
from SciGen.NeuralNetworks.FlexibleNetworks.FlexANN import FlexANN

class TestFlexANN(unittest.TestCase):
    def setUp(self):

        or_tups = [([0, 0], [0]), ([1, 0], [0]), ([0, 1], [0]), ([1, 1], [1])]
        self.orNet = FlexANN('l_int', or_tups, [([0, 0], [0]), ([1, 0], [1]), ([0, 1], [1]), ([1, 1], [1])],
                              [([0, 0], [0]), ([1, 0], [1]), ([0, 1], [1]), ([1, 1], [1])])

        xor_tups = [([0, 0], [0]), ([1, 0], [1]), ([0, 1], [0]), ([1, 1], [0])]
        self.xorNet = FlexANN('l_int', xor_tups, [([0, 0], [0]), ([1, 0], [1]), ([0, 1], [1]), ([1, 1], [0])],
                              [([0, 0], [0]), ([1, 0], [1]), ([0, 1], [1]), ([1, 1], [0])], complexity=2)

    def test_orGate(self):
        self.assertTrue(round(self.orNet.forward_propagate([0, 0])[0][0]) == 0.0)
        self.assertTrue(round(self.orNet.forward_propagate([0, 1])[0][0]) == 1.0)
        self.assertTrue(round(self.orNet.forward_propagate([1, 0])[0][0]) == 1.0)
        self.assertTrue(round(self.orNet.forward_propagate([1, 1])[0][0]) == 1.0)

    def test_xorGate(self):
        self.assertTrue(round(self.xorNet.forward_propagate([0, 0])[0][0]) == 0.0)
        self.assertTrue(round(self.xorNet.forward_propagate([0, 1])[0][0]) == 1.0)
        self.assertTrue(round(self.xorNet.forward_propagate([1, 0])[0][0]) == 1.0)
        self.assertTrue(round(self.xorNet.forward_propagate([1, 1])[0][0]) == 0.0)

unittest.main()

