# coding: utf-8

# Deep Neural Network for Image Classification: Application

# Packages

# - [numpy](www.numpy.org) is the fundamental package for scientific computing with Python.
# - [matplotlib](http://matplotlib.org) is a library to plot graphs in Python.
# - [h5py](http://www.h5py.org) is a common package to interact with a dataset that is stored on an H5 file.
# - [PIL](http://www.pythonware.com/products/pil/) and [scipy](https://www.scipy.org/) are used here to test your model with your own picture at the end.
# - dnn_app_utils provides the functions implemented.
# - np.random.seed(1) is used to keep all the random function calls consistent.
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils import *
from lr_utils import load_dataset

np.random.seed(1)


# dataset containing:
#     - a training set of m_train images labelled as cat (1) or non-cat (0)
#     - a test set of m_test images labelled as cat and non-cat
#     - each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB).

train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print("Number of training examples: " + str(m_train))
print("Number of testing examples: " + str(m_test))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_x_orig shape: " + str(train_x_orig.shape))
print("train_y shape: " + str(train_y.shape))
print("test_x_orig shape: " + str(test_x_orig.shape))
print("test_y shape: " + str(test_y.shape))

# Reshape and standardize the images before feeding them to the network. The code is given in the cell below.

# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T  # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.

print("train_x's shape: " + str(train_x.shape))
print("test_x's shape: " + str(test_x.shape))

# build a L layer Deep neural network to distinguish cat images from non-cat images.

# Deep Learning methodology to build the model:
#     1. Initialize parameters / Define hyperparameters
#     2. Loop for num_iterations:
#         a. Forward propagation
#         b. Compute cost function
#         c. Backward propagation
#         d. Update parameters (using parameters, and grads from backprop)
#     4. Use trained parameters to predict labels


# notice that running the model on fewer iterations (say 1500) gives better accuracy on the test set.
# This is called "early stopping" Early stopping is a way to prevent overfitting.

# L-layer Neural Network

# CONSTANTS
layers_dims = [12288, 20, 7, 5, 1]  # 5-layer model

# L_layer_model

def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):  # lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    """

    np.random.seed(1)
    costs = []  # keep track of cost

    # Parameters initialization.

    parameters = initialize_parameters_deep(layers_dims)


    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.

        AL, caches = L_model_forward(X, parameters)


        # Compute cost.

        cost = compute_cost(AL, Y)


        # Backward propagation.

        grads = L_model_backward(AL, Y, caches)


        # Update parameters.

        parameters = update_parameters(parameters, grads, learning_rate)


        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True)
print("Train data")
pred_train = predict(train_x, train_y, parameters)

print("Test data")
pred_test = predict(test_x, test_y, parameters)


# 5-layer neural network has better performance (80%) than your 2-layer neural network (72%) on the same test set.
