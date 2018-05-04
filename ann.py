import math
import numpy as np

class ArtificialNeuralNetwork:
    def __init__(self, dimensions):
        if not all(element > 0 for element in dimensions):
            raise Exception("Invalid input size")
        self.input_length  = dimensions[0]
        self.output_length = dimensions[-1]
        self.weights = [np.random.uniform(-1, 1, (dimensions[itr], dimensions[itr+1])) for itr in range(len(dimensions)-1)]

    def sigmoid(self, x):
        """ Sigmoid function """
        return 1/(1+math.e**(-x))

    def sigmoid_derivative(self, sigmoid_value):
        """ Derivative of activation function """
        return sigmoid_value*(1.0 - sigmoid_value)

    def activate(self, weights, inp):
        """ Activation function """
        return self.sigmoid(np.matmul(inp, weights))

    def forward_prop(self, inputs):
        """ Forward propagation """
        output, node_outputs = self._forward_prop(inputs)
        return output

    def _forward_prop(self, inputs):
        """ Forward propagation with output values for backpropagation"""
        if len(inputs) != self.input_length or type(inputs) not in (np.ndarray, list):
            raise Exception("Invalid input")
        elif type(inputs) is list:
            inputs = np.array(inputs)
        outputs = list()
        for itr in range(len(self.weights)):
            inputs = self.activate(self.weights[itr], inputs)
            outputs.append(inputs)
        return inputs, outputs

    def back_prop(self, inputs, exp_val, learning_rate=0.1):
        """ Backpropagation algorithm """
        if len(exp_val) != self.output_length or type(exp_val) not in (np.ndarray, list):
            raise Exception("Invalid expected value")
        elif type(exp_val) is list:
            exp_val = np.array(exp_val)
        weight_len = len(self.weights)
        obtained_val, outputs = self._forward_prop(inputs)
        outputs = np.array(outputs)
        error = (exp_val - outputs[-1]) * self.sigmoid_derivative(obtained_val) # initial error based on output
        delta = error * self.sigmoid_derivative(obtained_val) # initial delta
        deltas = [delta]
        for layer in range(weight_len-2, -1, -1): # calculate the error deltas for each set of weights
            output = self.sigmoid_derivative(outputs[layer])
            error = np.matmul(self.weights[layer+1], delta)
            delta = np.multiply(output, error)
            deltas.append(delta)
        for weights in range(weight_len): # update the weights based on error deltas
            curr_weights = self.weights[weight_len-weights-1]
            delta = np.resize(deltas[weights], (len(deltas[weights]), 1))
            weight_upd = np.matmul(delta, np.ones((1, len(curr_weights)))).T
            self.weights[weight_len-weights-1] = np.add(curr_weights, learning_rate*weight_upd)
