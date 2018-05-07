import math
import numpy as np

class ArtificialNeuralNetwork:
    def __init__(self, dimensions, weights=None):
        """ Instantiate Neural Network """
        if not all(element > 0 for element in dimensions):
            raise Exception("Invalid input size")
        self._input_length  = dimensions[0]
        self._output_length = dimensions[-1]
        # generate weights where for each matrix the rows represent the weights connecting to a node in the next layer
        if weights is None:
            self._weights = [np.random.uniform(-1, 1, (dimensions[itr+1], dimensions[itr])) for itr in range(len(dimensions)-1)]
        else:
            self._weights = self._load_weights(weights)
        self._biases = [np.random.uniform(-1, 1, (dimensions[itr+1])) for itr in range(len(dimensions)-1)]

    def _sigmoid(self, x):
        """ Sigmoid function """
        return 1/(1+math.e**(-x))

    def _sigmoid_derivative(self, sigmoid_value):
        """ Derivative of sigmoid function """
        return sigmoid_value*(1.0 - sigmoid_value)

    def _activate(self, weights, inp, weight_val):
        """ Activation function """
        return self._sigmoid(np.add(np.matmul(weights,inp), self._biases[weight_val]))

    def forward_prop(self, inputs):
        """ Forward propagation """
        value, outputs = self._forward_prop(inputs)
        return value

    def save(self, filename):
        """ Save the weights of your neural network """
        np.save(filename, self._weights)

    def _load_weights(self, filename):
        """ Private method: load weights into neural network """
        return np.load(filename)

    def _forward_prop(self, inputs):
        """ Forward propagation with output values for backpropagation"""
        if len(inputs) != self._input_length or type(inputs) not in (np.ndarray, list):
            raise Exception("Invalid input")
        elif type(inputs) is list:
            inputs = np.array(inputs)
        outputs = [inputs]
        for itr in range(len(self._weights)):
            inputs = self._activate(self._weights[itr], inputs, itr) # update values to propagate
            outputs.append(inputs) # keep track of outputs for backpropagation
        return inputs, np.array(outputs[:-1])

    def back_prop(self, inputs, exp_val, learning_rate=0.01):
        """ Backpropagation algorithm using matrix algebra """
        if len(exp_val) != self._output_length or type(exp_val) not in (np.ndarray, list):
            raise Exception("Invalid expected value -- check size or type")
        elif type(exp_val) is list:
            exp_val = np.array(exp_val)
        ret_val, output_values = self._forward_prop(inputs) # forward prop values
        ret_val = np.resize(ret_val,(len(ret_val),1)) # resize into vector format
        hidden = np.resize(output_values[-1],(len(output_values[-1]),1)) # resize into vector format
        targets = np.resize(exp_val,(len(exp_val), 1)) # resize into vector format
        error = np.add(targets, -1 * ret_val) # generate initial error
        gradients = np.multiply(self._sigmoid_derivative(ret_val), error) * learning_rate # get initial gradients
        weight_deltas = np.matmul(gradients, hidden.T) # generate initial weight deltas
        self._biases[-1] = np.add(self._biases[-1], gradients)
        self._weights[-1] = np.add(self._weights[-1], weight_deltas) # update weights based on weight deltas
        for itr in reversed(range(len(self._weights))[1:]):
            # resize into vector format and update hidden layer weights
            inputs = np.resize(output_values[itr-1], (len(output_values[itr-1]), 1))
            weight_m = self._weights[itr].T # generate transpose of corresponding weight matrix
            error = np.matmul(weight_m, error) # calculate error
            gradients = np.multiply(self._sigmoid_derivative(hidden), error)*learning_rate # calculate gradient
            self._biases[itr-1] = np.add(self._biases[itr-1], np.resize(gradients, (len(self._biases[itr-1]))))
            weight_deltas = np.matmul(gradients, inputs.T) # generate weight deltas
            self._weights[itr-1] = np.add(self._weights[itr-1], weight_deltas) # update weights
            hidden = np.resize(output_values[itr-1], (len(output_values[itr-1]), 1)) # update hidden layer outputs

        