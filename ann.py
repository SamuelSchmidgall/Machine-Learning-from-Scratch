import math
import numpy as np

class ArtificialNeuralNetwork:
    def __init__(self, dimensions):
        if not all(element > 0 for element in dimensions):
            raise Exception("Invalid input size")
        self.input_length  = dimensions[0]
        self.output_length = dimensions[-1]
        self.weights = [np.random.uniform(-1, 1, (dimensions[itr+1], dimensions[itr])) for itr in range(len(dimensions)-1)]


    def sigmoid(self, x):
        """ Sigmoid function """
        return 1/(1+math.e**(-x))

    def sigmoid_derivative(self, sigmoid_value):
        """ Derivative of activation function """
        return sigmoid_value*(1.0 - sigmoid_value)

    def activate(self, weights, inp):
        """ Activation function """
        return self.sigmoid(np.matmul(weights,inp))

    def forward_prop(self, inputs):
        """ Forward propagation """
        a, b = self._forward_prop(inputs)
        return a

    def _forward_prop(self, inputs):
        """ Forward propagation with output values for backpropagation"""
        if len(inputs) != self.input_length or type(inputs) not in (np.ndarray, list):
            raise Exception("Invalid input")
        elif type(inputs) is list:
            inputs = np.array(inputs)
        outputs = [inputs]
        for itr in range(len(self.weights)):
            inputs = self.activate(self.weights[itr], inputs)
            outputs.append(inputs)
        return inputs, np.array(outputs[:-1])

    def back_prop(self, inputs, exp_val, learning_rate=0.01):
        """ Backpropagation algorithm """
        if len(exp_val) != self.output_length or type(exp_val) not in (np.ndarray, list):
            raise Exception("Invalid expected value")
        elif type(exp_val) is list:
            exp_val = np.array(exp_val)
        ret_val, output_values = self._forward_prop(inputs)
        ret_val = np.resize(ret_val,(len(ret_val),1))
        hidden = np.resize(output_values[-1],(len(output_values[-1]),1))
        targets = np.resize(exp_val,(len(exp_val), 1))
        error = np.add(targets, -1 * ret_val)
        gradients = np.multiply(self.sigmoid_derivative(ret_val), error) * learning_rate
        weight_ho_deltas = np.matmul(gradients, hidden.T)
        self.weights[-1] = np.add(self.weights[-1], weight_ho_deltas)
        for itr in reversed(range(len(self.weights))[1:]):
            inputs = np.resize(output_values[itr-1], (len(output_values[itr-1]), 1))
            who_t = self.weights[itr].T
            error = np.matmul(who_t, error)
            gradients = np.multiply(self.sigmoid_derivative(hidden), error)*learning_rate
            weight_ih_deltas = np.matmul(gradients, inputs.T)
            self.weights[itr-1] = np.add(self.weights[itr-1], weight_ih_deltas)
            hidden = np.resize(output_values[itr-1], (len(output_values[itr-1]), 1))








ann = ArtificialNeuralNetwork([2, 4, 2, 5, 2])
for i in range(10000):
    ann.back_prop(np.array([1,1]), np.array([0,0]))

