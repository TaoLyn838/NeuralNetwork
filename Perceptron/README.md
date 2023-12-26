# Simple Perceptron Neural Network

## Overview

The Simple Perceptron Neural Network is a Python-based implementation of the fundamental perceptron, a building block of artificial neural networks. This project demonstrates the core concepts of neural network training, including weight initialization, activation functions, and the learning process through weight adjustments.

## Features

- **Singular Neuron Model:** Implements a single-layer, single-neuron model, embodying the basic principles of neural networks.
- **Customizable Training:** Users can specify their own data, learning rate, and training epochs.
- **He Initialization:** Utilizes He et al.'s weight initialization method for effective training performance.
- **Activation Function:** Includes a simple, threshold-based activation function to determine neuron output.

## How It Works

The perceptron receives input data and applies a set of weights to these inputs. The weighted inputs are then summed up and passed through an activation function, which is threshold-based in this implementation. Based on the output of the activation function, the perceptron adjusts its weights to reduce the prediction error. This process is repeated for a specified number of epochs or until the error reaches an acceptable level.

## Walkthrough
See simple output in jupyter file [simplePerceptron.ipynb](https://github.com/TaoLyn838/NeuralNetwork/blob/main/Perceptron/simplePerceptron.ipynb)
![output](https://github.com/TaoLyn838/NeuralNetwork/assets/58400041/c4d36b3f-ecee-4331-bc55-764494374d26)
