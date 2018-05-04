import math
import numpy as np

class ArtificialNeuralNetwork:
    def __init__(self, dimensions):
        if len(dimensions) <= 2:
            raise Exception("Input size too small")
        self.input_length  = dimensions[0]
        self.output_length = dimensions[-1]
        self.weights = [np.random.uniform(-1, 1, (dimensions[itr], dimensions[itr+1])) for itr in range(len(dimensions)-1)]

    def sigmoid(self, x):
        return 1/(1+math.e**(-x))

    def sigmoid_derivative(self, sigmoid_value):
        return sigmoid_value*(1.0 - sigmoid_value)

    def activate(self, weights, inp):
        return self.sigmoid(np.matmul(inp, weights))

    def forward_prop(self, inputs):
        if len(inputs) != self.input_length:
            raise Exception("Input size too small")
        outputs = list()
        for itr in range(len(self.weights)):
            inputs = self.activate(self.weights[itr], inputs)
            outputs.append(inputs)
        return inputs, outputs

    def back_prop(self, inputs, exp_val, learning_rate=0.1):
        weight_len = len(self.weights)
        obtained_val, outputs = self.forward_prop(inputs)
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
            weight_upd = np.matmul(delta, np.ones((1, len(curr_weights)))).T # err
            self.weights[weight_len-weights-1] = np.add(curr_weights, learning_rate*weight_upd)


