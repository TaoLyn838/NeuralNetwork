import numpy as np


class SimplePerceptron:
    def train(self):


        return self.weights

    def activate(self):


    def __init__(self, data, threshold):
        """
        :param data (list): The sequence of input as X1, X2, …Xn
        """
        self.X_n = data
        self.threshold = threshold

        # He et al's weights initialization
        self.weights = np.random.randn(1, len(data)) * np.sqrt(2 / len(data))[0]
