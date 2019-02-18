# SciGen
## Scientific computing library built from scratch that contains many modern data science tools as well as a few models that automatically generate based on the given dataset.
 
## Features:
  * Artificial Neural Network
  * K-Nearest Neighbors
  * Logistic Regression
  * Simple Linear Regression
  * Multiple Linear Regression
  * K-Means Clustering
  * Classification Tree
  
## Features to be implemented:
  * Smoothing Splines
  * Random Forests
  * Cross-Validation Support
  * Local Regression
  * Generalized Additive Models
  * Regression Trees
  * Dimension Reduction Methods
  * Maximal Margin Classifier
  * Support Vector Classifier
  * Support Vector Machines
  * Convolutional Neural Network
  * Recurrent Neural Network
  * Principle Component Analysis
  * LSTM Neural Network
  * Neuro-evolution
  * Classic Evolutionary Algorithm


## Using SciGen:

* Example of Deep Reinforcement Learning with Neural Network
```python
import numpy as np
from Network.layers import Linear
from random import randint, choice, uniform
from Network.neural_network import NeuralNetwork
from Network.optimizers import MSEStochasticGradientDescent

""" Creating the model """
class CartPolePolicyNetwork:
    def __init__(self):
        self._GAMMA = 0.999
        self._batch_size = 100
        self._actions = [0, 1]
        self._learning_modulus = 100
        self._memory = StateMemory(10000)
        self._random_action_probability = 0.15
        self._architecture = [Linear(4, 8, dropout_probability=0.1), Linear(8, 8, dropout_probability=0.2), Linear(8, len(self._actions))]
        self._net = NeuralNetwork(self._architecture, MSEStochasticGradientDescent(), minibatch_size=4)

    def action(self, state, train=False):
        prediction = self._net.predict(state)
        if train and uniform(0, 1) <= self._random_action_probability:
            return choice(self._actions)
        return self._actions[int(np.argmax(prediction))]

    def fit(self, action, current_state, previous_state, reward, iteration):
        self._memory.append((action, current_state, previous_state, reward))
        if iteration%self._learning_modulus == 0 and len(self._memory) >= self._batch_size:
            _batch_X, _batch_Y = list(), list()
            _sample = self._memory.sample(self._batch_size)
            for _elem in _sample:
                _s_action, _s_curr_state, _s_prev_state, _s_reward = _elem
                _y_exp = np.zeros(len(self._actions))
                if _s_reward != 1.5:
                    _y_exp[1 if _s_action == 0 else 0] = 1
                else:
                    _y_exp[_s_action] = 1
                _batch_X.append(_s_prev_state)
                _batch_Y.append(_y_exp)
            self._net.fit(_batch_X, _batch_Y, 1)
```

