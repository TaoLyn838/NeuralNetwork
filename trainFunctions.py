import numpy as np


def train_test_split(X, Y, test_size=0.2):
    test_size = int(X.shape[0] * test_size)

    x = np.array(X)
    y = np.array(Y)

    x_train = x[:-test_size]
    x_test = x[-test_size:]
    y_train = y[:-test_size]
    y_test = y[-test_size:]
    return x_train, x_test, y_train, y_test


def random_train_test_split(X, Y, test_size_percentage=0.2):
    test_size = int(X.shape[0] * test_size_percentage)

    x = np.array(X)
    y = np.array(Y)

    # Randomize data
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    x_train = x[:-test_size]
    x_test = x[-test_size:]
    y_train = y[:-test_size]
    y_test = y[-test_size:]
    return x_train, x_test, y_train, y_test
