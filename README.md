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

* Artificial Neural network
```python
from SciGen.NeuralNetworks.Artificial import ArtificialNeuralNetwork

""" Creating the model """
stock_net = ArtificialNeuralNetwork(dimensions=[1, 10, 15, 2])

""" Obtaining model data """
stock_data = [([0.3, 0.423, ..], 0.12), ([0.04, -0.2, ...], -0.04), ...]  # ([daily_cost_delta, weekly_cost_delta, ...], expected_return_percentage)

""" Training the model """
for data in stock_data:
    inp, expected = data[0], data[1]
    stock_net.back_prop(inp, expected)

""" Model predictions """
stock_data = [0.32, 0.56, ...]  # [daily_cost_delta, weekly_cost_delta, ...]
model_prediction = stock_net.predict(stock_data)
```

