# -*- coding: utf-8 -*-
"""
TRAIN neural network to classify two Gaussian-distributed clusters in 2d space.

Toy example from:
http://nbviewer.jupyter.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb
"""
#from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
##########################
##########################
import os
os.chdir("C:\\Users\\Raul Vazquez\\Desktop\\reSkilling\\reSkilling\\nn_implementation")
############################
############################
from nn2_theanoClass import MLP
from utils_loc import gradient_updates_momentum
import utils_loc
# ====================== GENERATE TRAINING DATA ===============================
# Training data - two randomly-generated Gaussian-distributed clouds of points in 2d space
np.random.seed(0)
theano.config.floatX = 'float32'
# Number of points
N = 1000
# Labels for each cluster
y = np.random.randint(0, 2, N)
# Mean of each cluster
means = np.array([[-1, 1], [-1, 1]])
# Covariance (in X and Y direction) of each cluster
covariances = np.random.random_sample((2, 2)) + 1
# Dimensions of each point
X = np.vstack([np.random.randn(N)*covariances[0, y] + means[0, y],
               np.random.randn(N)*covariances[1, y] + means[1, y]]).astype(theano.config.floatX)
# Convert to targets, as floatX
y = y.astype(theano.config.floatX)
# Plot the data
plt.figure(figsize=(8, 8))
plt.scatter(X[0, :], X[1, :], c=y, lw=.3, s=3, cmap=plt.cm.cool)
plt.axis([-6, 6, -6, 6])
plt.show()


# =================== INITIALIZE ===========================
'''
# Set the size of each layer (and the number of layers)
#   INPUT layer size is training data dimensionality (1.e, 2)
#   OUTPUT layer size is just 1-d: class label: 0 or 1
#   Let HIDDEN layers be twice the size of the input.
'''
# If we wanted more layers, we could just add another layer size to this list.
layer_sizes = [X.shape[0], X.shape[0]*2, 1] # this one has 2-inputs, 4-hidden and 1-output units

# INITIALIZE PARAMETERS
W_init = []
b_init = []
activations = []
for n_input, n_output in zip(layer_sizes[:-1], layer_sizes[1:]):
    W_init.append(np.random.randn(n_output, n_input)/ np.sqrt(n_input))
    b_init.append(np.zeros(n_output))
    # SIGMOID activation for all layers
    activations.append(T.nnet.sigmoid)


# Create an instance of the MLP class
mlp = MLP(W_init, b_init, activations)


# Create Theano variables for the MLP input
mlp_input = T.matrix('mlp_input')
# the desired output
mlp_target = T.vector('mlp_target')


learning_rate = 0.01
momentum = 0.9
# Create a function for computing the cost of the network given an input
cost = mlp.squared_error(mlp_input, mlp_target)
# Create a theano function for training the network
'''theano.function([input], obj_fun, updates=fn_to_udate_shared_vars))'''
train = theano.function([mlp_input, mlp_target], cost,
                        updates=gradient_updates_momentum(cost, mlp.params, learning_rate, momentum))
# Create a theano function for computing the MLP's output given some input
mlp_output = theano.function([mlp_input], mlp.output(mlp_input))



iterat = 0
max_iter = 20
'''TO_DO: implement "early stopping":
    use a hold-out validation set. When the validation error starts to increase,
    stop training the net because the network is overfitting.  
'''
while iterat < max_iter:
    current_cost = train(X, y)
    current_output = mlp_output(X)
    accuracy = np.mean((current_output > .5) == y)
    # Plot network output after each iteration
    '''plt.figure(figsize=(8, 8))
    plt.scatter(X[0, :], X[1, :], c=current_output>.5,
                lw=.3, s=3, cmap=plt.cm.cool, vmin=0, vmax=1)
    plt.axis([-6, 6, -6, 6])
    plt.title('Cost: {:.3f}, Accuracy: {:.3f}'.format(float(current_cost), accuracy))
    plt.show()
    iterat += 1'''
    
    
utils_loc.plot_decision_boundary( (lambda x: utils_loc.predict(W1, b1, W2, b2, x)), X, y )
plt.title("Decision Boundary of initial Parameters")

utils_loc.plot_decision_boundary( (lambda x: utils_loc.predict_sigmoid(W_init[0].T, b_init[0].T, W_init[1].T, b_init[1].T, x)), X.T, y)

plt.title("Decision Boundary of initial Parameters")