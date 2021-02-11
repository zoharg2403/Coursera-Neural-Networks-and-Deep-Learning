
import numpy as np

#### Helper functions ####

def layer_sizes(X, Y):
    """
    Arguments:
            X -- input dataset of shape (input size, number of examples)
            Y -- labels of shape (output size, number of examples)
    Returns:
            n_x -- the size of the input layer
            n_h -- the size of the hidden layer
            n_y -- the size of the output layer
    """
    n_x = np.shape(X)[0]  # size of input layer
    n_h = 4  # set to 4 (4 nodes / neurons in the hidden layer)
    n_y = np.shape(Y)[0]  # size of output layer
    return n_x, n_h, n_y

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
            n_x -- size of the input layer
            n_h -- size of the hidden layer
            n_y -- size of the output layer
    Returns:
            python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def forward_propagation(X, parameters):
    """
    Argument:
            X -- input data of size (n_x, m)
            parameters -- python dictionary containing your parameters (output of initialize_parameters function
            or update_parameters function)
    Returns:
            A2 -- The output of the second activation
            python dictionary containing: "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)  # tanh activation function
    Z2 = np.dot(W2, A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))  # sigmoid activation function
    assert (A2.shape == (1, X.shape[1]))
    return A2, {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

def compute_cost(A2, Y, parameters):
    """
    Arguments:
            A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
            Y -- "true" labels vector of shape (1, number of examples)
            parameters -- python dictionary containing your parameters (output of initialize_parameters function)
    Returns:
            cost -- cross-entropy cost
    """
    m = Y.shape[1]  # number of example
    # Compute the cross-entropy cost
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
    cost = - np.sum(logprobs) / m
    cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect (turns [[#]] into # (type float))
    assert (isinstance(cost, float))
    return cost

def backward_propagation(parameters, cache, X, Y):
    """
    Arguments:
            parameters -- python dictionary containing our parameters (W1, b1, W2, b2)
            cache -- a dictionary containing "Z1", "A1", "Z2" and "A2" (output of forward_propagation).
            X -- input data of shape (2, number of examples)
            Y -- "true" labels vector of shape (1, number of examples)
    Returns:
            grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]  # number of examples
    # retrieve W2 from the dictionary "parameters".
    W2 = parameters['W2']
    # Retrieve A1 and A2 from dictionary "cache".
    A1 = cache['A1']
    A2 = cache['A2']
    # Backward propagation: calculate dW1, db1, dW2, db2.
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.tanh(cache['Z1']) ** 2)
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

def update_parameters(parameters, grads, learning_rate=1.2):
    """
    Arguments:
            parameters -- python dictionary containing your parameters (output of forward_propagation function)
            grads -- python dictionary containing your gradients (output of backward_propagation function)
    Returns:
            parameters -- python dictionary containing your updated parameters
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    # Update rule for each parameter
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters

#### NN model ####

def nn_model(X, Y, num_iterations=10000, print_cost=False):
    """
    Arguments:
            X -- dataset of shape (2, number of examples)
            Y -- labels of shape (1, number of examples)
            n_h -- size of the hidden layer
            num_iterations -- Number of iterations in gradient descent loop
            print_cost -- if True, print the cost every 1000 iterations
    Returns:
            parameters -- parameters learnt by the model. They can then be used to predict.
    """
    # Initialize sizes
    n_x, n_h, n_y = layer_sizes(X, Y)
    # Initialize parameters
    parameters = initialize_parameters(n_x, n_h, n_y)
    # Loop - gradient descent
    for i in range(0, num_iterations):
        # Forward propagation:
        A2, cache = forward_propagation(X, parameters)
        # Cost function:
        cost = compute_cost(A2, Y, parameters)
        # Backpropagation:
        grads = backward_propagation(parameters, cache, X, Y)
        # Gradient descent parameter update:
        parameters = update_parameters(parameters, grads)
        # Print the cost every 1000 iterations:
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
    return parameters


#### Prediction  ####

def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
        Arguments:
                parameters -- python dictionary containing your parameters (output of the nn model)
                X -- input data of size (n_x, m)
        Returns
                predictions -- vector of predictions of our model
    """
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = forward_propagation(X, parameters)  # cache containes Z1 A1 Z2 and A2
    predictions = (A2 > 0.5)
    return predictions


