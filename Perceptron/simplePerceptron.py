import numpy as np


class SimplePerceptron:
    def train(self, epochs):
        predict_list = []
        for _ in range(epochs):
            predict = np.dot(self.data, self.weights)
            predict_list.append(predict)

            output = self.activate(predict)
            if output < 0:
                self.weights = self.weights + self.learning_rate * self.data
            elif output > 0:
                self.weights = self.weights - self.learning_rate * self.data
            else:
                break

        return predict_list

    def activate(self, predict):
        if predict > self.threshold:
            return 1
        elif predict < self.threshold:
            return -1
        return 0

    def __init__(self, data, threshold=0, learning_rate=0.0001):
        """
        :param data (list): The sequence of input as X1, X2, …Xn
        """
        self.data = data
        self.num_inputs = len(self.data)
        self.threshold = threshold
        self.learning_rate = learning_rate

        # Use He et al's weights initialization to set random w_j
        self.weights = np.random.rand(self.num_inputs) * np.sqrt(2 / self.num_inputs)
