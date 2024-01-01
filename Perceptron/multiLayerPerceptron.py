import numpy as np
import matplotlib.pyplot as plt

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
    def train(self, x_train, x_test, y_train, y_test, epoch=100, threshold=1000):
        train_predicts = None
        train_loss = None
        test_predicts = None
        test_loss = None
        x_points = []
        x2_points = []
        y_points = []

        for i in range(epoch):
            train_predicts = self.forward_propagation(x_train)

            train_loss = self.backward_propagation(x_train, y_train)

            if i % threshold == 0:
                test_predicts = self.forward_propagation(x_test)
                test_loss = self.mean_squared_err(test_predicts, y_test)

                train_accuracy = self.calculate_accuracy(train_predicts, y_train)
                test_accuracy = self.calculate_accuracy(test_predicts, y_test)
                x_points.append(train_accuracy)
                x2_points.append(test_accuracy)
                y_points.append(i)
                print(f"Iteration: {i}, Training Loss: {train_loss}, Test Loss: {test_loss}. "
                      f"Train accuracy: {train_accuracy}, Test accuracy: {test_accuracy}")
            # else:
            #     print(f"Iteration: {i}, Training Loss: {loss}")

        plt.figure(figsize=(10, 5))
        plt.plot(y_points, x_points, label="Train accuracy", marker='o', linestyle='-', color='blue')
        plt.plot(y_points, x2_points, label="Test accuracy", marker='x', linestyle='-', color='green')
        plt.title("Train accuracy vs. Test accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid = True
        plt.legend()
        plt.show()

        return train_predicts, train_loss, test_predicts, test_loss

    def relu_activated(self, x):
        return np.maximum(0, x)

    # def softmax_activated(self, predicts):
    #     exps = np.exp(predicts - np.max(predicts, axis=1, keepdims=True))
    #     return exps / np.sum(exps, axis=1, keepdims=True)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def one_hot_encode(self, true_labels):
        # get total number of classes
        classes = self.output_layer.shape[1]

        return np.eye(classes)[true_labels]

    def mean_squared_err(self, predicts, actual_labels):
        one_hot_actual_labels = self.one_hot_encode(actual_labels)
        difference = predicts - one_hot_actual_labels
        individual_errors = np.power(difference, 2)
        # overall_mse = np.sum(individual_errors) / predicts.shape[0]

        return np.mean(individual_errors)

    def mse_derivative(self, predicts, one_hot_actual_labels):
        return 2 * (predicts - one_hot_actual_labels) / predicts.shape[0]

    def forward_propagation(self, X):
        # Calculate layers
        self.h1 = np.dot(X, self.w1) + self.b1
        self.h1_output = self.relu_activated(self.h1)
        self.h2 = np.dot(self.h1_output, self.w2) + self.b2
        self.h2_output = self.relu_activated(self.h2)
        self.h3 = np.dot(self.h2_output, self.w3) + self.b3
        self.h3_output = self.relu_activated(self.h3)
        self.h4 = np.dot(self.h3_output, self.w4) + self.b4
        self.output_layer = self.relu_activated(self.h4)

        return self.output_layer

    def backward_propagation(self, X, actual_labels):
        mse_err = self.mean_squared_err(self.output_layer, actual_labels)
        output_delta = self.mse_derivative(self.output_layer, self.one_hot_encode(actual_labels))
        h3_err = np.dot(output_delta, self.w4.T)
        h3_delta = h3_err * self.relu_derivative(self.h3)
        h2_err = np.dot(h3_delta, self.w3.T)
        h2_delta = h2_err * self.relu_derivative(self.h2)
        h1_err = np.dot(h2_delta, self.w2)
        h1_delta = h1_err * self.relu_derivative(self.h1)

        # # Calculate gradients(∂c/∂ω) for weights and biases
        d_w4 = np.dot(self.h3_output.T, output_delta)
        d_w3 = np.dot(self.h2_output.T, h3_delta)
        d_w2 = np.dot(self.h1_output.T, h2_delta)
        d_w1 = np.dot(X.T, h1_delta)
        d_b4 = np.sum(output_delta, axis=0)
        d_b3 = np.sum(h3_delta, axis=0)
        d_b2 = np.sum(h2_delta, axis=0)
        d_b1 = np.sum(h1_delta, axis=0)

        # Update weights and biases
        self.w4 -= self.learning_rate * d_w4
        self.w3 -= self.learning_rate * d_w3
        self.w2 -= self.learning_rate * d_w2
        self.w1 -= self.learning_rate * d_w1
        self.b4 -= self.learning_rate * d_b4
        self.b3 -= self.learning_rate * d_b3
        self.b2 -= self.learning_rate * d_b2
        self.b1 -= self.learning_rate * d_b1

        return mse_err
        # return h3_err

    def calculate_accuracy(self, predictions, actual_labels):
        predicted_classes = np.argmax(predictions, axis=1)
        correct_predictions = np.sum(predicted_classes == actual_labels)
        accuracy = correct_predictions / len(actual_labels)
        return accuracy

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.0001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights for input -> hidden layer and hidden -> output layer
        self.w1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2 / self.hidden_size)
        self.w2 = np.random.randn(self.hidden_size, self.hidden_size) * np.sqrt(2 / self.hidden_size)
        self.w3 = np.random.randn(self.hidden_size, self.hidden_size) * np.sqrt(2 / self.hidden_size)
        self.w4 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2 / self.hidden_size)

        # Initialize biases for hidden and output layers
        self.b1 = np.zeros((1, self.hidden_size))
        self.b2 = np.zeros((1, self.hidden_size))
        self.b3 = np.zeros((1, self.hidden_size))
        self.b4 = np.zeros((1, self.output_size))
