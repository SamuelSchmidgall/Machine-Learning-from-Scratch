import numpy as np


class NeuralNetwork:
    """ Neural Network Architecture """
    def __init__(self, layers, optimizer, minibatch_size=100, learning_rate=0.001):
        """ Initialization function for Neural Network class """
        self._layers = layers
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._minibatch_size = minibatch_size # size of minibatch samples
        # add widen and deepen functionality

    def predict(self, x):
        """ Predict corresponding y_hat with respect to x """
        for layer in self._layers:
            x = layer.forward(x)
        return x

    def fit(self, X, Y, iterations=1000):
        """ Fit vectors x elem of X with respect to corresponding y elem of Y """
        _mini_batches_x = [[x for x in X[_b*self._minibatch_size:(_b+1)*self._minibatch_size]] for _b in range(len(X)//self._minibatch_size + 1)]
        _mini_batches_y = [[y for y in Y[_b*self._minibatch_size:(_b+1)*self._minibatch_size]] for _b in range(len(Y)//self._minibatch_size + 1)]
        for _iterations in range(iterations):
            for _batch in range(len(_mini_batches_x)):
                _batch_weight_gradients = self._compute_weight_gradients(_mini_batches_x, _mini_batches_y, _batch)
                self._update_weights(_batch_weight_gradients)
                _batch_weight_gradients.clear()

    def resize(self, layers):
        """ Use transfer learning to resize neural network """

    def _compute_weight_gradients(self, _mini_batches_x, _mini_batches_y, _batch):
        """ Compute weight gradients for respective layers """
        _batch_weight_gradients = list()
        for _elem in range(len(_mini_batches_x[_batch])):
            _propagated_values = list()
            x, y = _mini_batches_x[_batch][_elem], _mini_batches_y[_batch][_elem]
            for _layer in self._layers:
                _propagated_values.append(x)
                x = _layer.forward(x)
            _batch_weight_gradients.append(self._optimizer.compute_gradients(self._layers, _propagated_values, y, x))
            _propagated_values.clear()
        return _batch_weight_gradients

    def _update_weights(self, _batch_weight_gradients):
        """ Update the weights of each layer in the network based on computed gradients """
        for _weight_gradient in _batch_weight_gradients:
            _weight_gradient = list(reversed(_weight_gradient))
            for _layer in reversed(range(len(self._layers))):
                self._layers[_layer].update_weights(-self._learning_rate*_weight_gradient[_layer])



















