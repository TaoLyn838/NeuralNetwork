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
    def __init__(self,  input_size, hidden_size, output_size, learning_rate=0.0001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights for input -> hidden layer and hidden -> output layer
        self.w1 = np.random.randn


