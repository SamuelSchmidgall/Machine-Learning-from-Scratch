import unittest
from FlexibleNetworks.FlexANN import FlexANN

class TestFlexANN(unittest.TestCase):
    def setUp(self):
        and_tups = [([0, 0], [0]), ([1, 0], [0]), ([0, 1], [0]), ([1, 1], [1])]
        self.andNet = FlexANN('l_int', and_tups, [([0, 0], [0]), ([1, 0], [0]), ([0, 1], [0]), ([1, 1], [1])],
                                                 [([0, 0], [0]), ([1, 0], [0]), ([0, 1], [0]), ([1, 1], [1])])
        or_tups = [([0, 0], [0]), ([1, 0], [0]), ([0, 1], [0]), ([1, 1], [1])]
        self.orNet = FlexANN('l_int', or_tups, [([0, 0], [0]), ([1, 0], [1]), ([0, 1], [1]), ([1, 1], [1])],
                              [([0, 0], [0]), ([1, 0], [1]), ([0, 1], [1]), ([1, 1], [1])])

        xor_tups = [([0, 0], [0]), ([1, 0], [1]), ([0, 1], [0]), ([1, 1], [0])]
        self.xorNet = FlexANN('l_int', xor_tups, [([0, 0], [0]), ([1, 0], [1]), ([0, 1], [1]), ([1, 1], [0])],
                              [([0, 0], [0]), ([1, 0], [1]), ([0, 1], [1]), ([1, 1], [0])])

    def test_andGate(self):
        self.assertTrue(round(self.andNet.forward_propagate([0,0])[0][0]) == 0.0)
        self.assertTrue(round(self.andNet.forward_propagate([0,1])[0][0]) == 0.0)
        self.assertTrue(round(self.andNet.forward_propagate([1,0])[0][0]) == 0.0)
        self.assertTrue(round(self.andNet.forward_propagate([1,1])[0][0]) == 1.0)

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

