import numpy as np
import pandas as pd

"""
Why we need to use Multilayer perceptron:  In 1969, Minsky and Papert highlights that 
    "the fact that Perceptron, with only one neuron, can’t be applied to non-linear data."
    And, "The Multilayer Perceptron was developed to tackle this limitation. It is a neural network where the 
    mapping between inputs and output is non-linear." (Carolina Bento)

An algorithm that classifies input by separating two categories with a straight line. 
    Input is typically a feature vector x multiplied by weights w and added to a bias b: y = w * x + b. (Jorge Leonel)

The bias can be thought of as how much flexible the perceptron is. It is somehow similar to the constant
    b of a linear function y = ax + b. It allows us to move the line up and down to fit the prediction 
    with the data better. Without b the line will always goes through the origin (0, 0) and you may get 
    a poorer fit. (Jorge Leonel)
"""


class MultiLayerPerceptron:
    def relu_actived(self, x):
        return np.maximum(0, x)
    def forward_propagation(self, X):
        self.h1 = np.dot(X, self.w_input) + self.b1
        self.h1_output = self.relu_actived(self.h1)






    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.0001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights for input -> hidden layer and hidden -> output layer
        self.w_input = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2 / self.hidden_size)
        self.w1_hidden = np.random.randn(self.hidden_size, self.hidden_size) * np.sqrt(2 / self.hidden_size)
        self.w2_hidden = np.random.randn(self.hidden_size, self.hidden_size) * np.sqrt(2 / self.hidden_size)
        self.w3_hidden = np.random.randn(self.hidden_size, self.hidden_size) * np.sqrt(2 / self.hidden_size)
        self.w_output = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2 / self.output_size)

        # Initialize biases for hidden and output layers
        self.b1 = np.zeros((1, self.hidden_size))
        self.b2 = np.zeros((1, self.hidden_size))
        self.b3 = np.zeros((1, self.hidden_size))
        self.b4 = np.zeros((1, self.hidden_size))
        self.b5 = np.zeros((1, self.output_size))
