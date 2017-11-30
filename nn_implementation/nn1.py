# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:13:56 2017

@author: Raul Vazquez
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import scipy.optimize as op

# Generate a dataset and plot it
np.random.seed(0)
X, y = datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)


'''
Build a 3-layer neural network with 1 input layer, 1 hidden layer and 1 output layer. 
    The number of nodes in the input  layer is 2; the dimensionality of our data.
    The number of nodes in the output layer is 2; the number of classes we have.
    The number of nodes in the hidden layer will vary.

MODELS:
 1) tanh
    FWD:
        # INPUT layer activation values
        a1 = X.copy() 
        # HIDDEN layer activation values
        z2 = a1.dot(W1) + b1
        a2 = np.tanh(z2)
        # OUTPUT layer activation values
        z3 = a2.dot(W2) + b2
        a3 = softmax(z3) = np.exp(z3) / np.sum(exp_scores, axis=1, keepdims=True)
    BackWD:
        # output errors
        delta3 = a3 - y
        # hidden layer errors
        delta2 = (delta3.dot( W2')) * (1 - tanh**2(z2)) = (delta3.dot( W2')) * (1 - a2.^2)
        #derivatives
        dW2 = a2' * delta3
        db2 = delta3 # bias unit derivative wrt loss-func, hence change of 1
        dW1 = a1' * delta2
        db1 = delta2            
 2) sigmoid
    FWD:
        # INPUT layer activation values
        a1 = X.copy() 
        # HIDDEN layer activation values
        z2 = a1.dot(W1) + b1
        a2 = utils_loc.sigmoid(z2)
        # OUTPUT layer activation values
        z3 = a2.dot(W2) + b2
        a3 = utils_loc.sigmoid(z3) 
        
'''

num_examples = len(X) # training set size
m = len(X)
nn_input_dim = X.shape[1] # input layer dimensionality
nn_output_dim = 2 # output layer dimensionality
nn_hdim = 4 # hidden layer dmensionality
 
# Gradient descent parameters
epsilon = 0.01 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength

# Initialize the parameters to random values. We need to learn these.
np.random.seed(0)
W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
b1 = np.zeros((1, nn_hdim))
W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
b2 = np.zeros((1, nn_output_dim))

def nn_costFn_TanH(X, y, W1, b1, W2, b2, reg_lambda):
    '''
    Forward propagation to compute the value of the loss function
    Backward propagation to compute the value of the gradient of the loss function 
    '''
    
    """TO DO:
        Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
    """
    m = len(X)
    #FWD
    a1 = X.copy()
    z2 = a1.dot(W1) + b1
    a2 = np.tanh(z2)
    z3 = a2.dot(W2) + b2
    exp_scores = np.exp(z3)
    a3 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    ''' Recall that each row of a3 contains the probabilities of belonging to each class '''    
    # Calculating the loss
    ''' this is the same than computing the:
        Loss = -(1/m) * sum_1^m(sum_1^K (y[i,k] * log(a3[i,k]) + (1 - y[i,k]) * log(1 - a3[i,k])  ))
        for Y being a matrix representation of y with boolean categorical rows 
    '''
    corect_logprobs = -np.log(a3[range(m), y]) # choose the probability of having the observed output for each example
    data_loss = 1/m * np.sum(corect_logprobs)
    # Add regulatization term to the loss
    data_loss += reg_lambda/(2*m) * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    
    # BackWD
    delta3 = a3.copy()
    delta3[range(m), y] -= 1 # recall delta3 = a3 - y, when y defd as a matrix with boolean canonical entries
    delta2 = delta3.dot(W2.T) * (1 - np.power(a2, 2))
    # derivatives
    dW2 = (a2.T).dot(delta3)
    db2 = np.sum(delta3, axis=0, keepdims=True)
    dW1 = np.dot(a1.T, delta2)
    db1 = np.sum(delta2, axis=0)
 
    # Add regularization terms (b1 and b2 are bias terms: don't have regularization terms)
    dW2 += reg_lambda * W2
    dW1 += reg_lambda * W1
 
    # unroll the gradients
    grad = np.append(np.append(db1, (dW1.T).ravel()), np.append(db2, (dW2.T).ravel() ) )
    return (data_loss, grad)




'''
if 'jac' is a boolean and is True, 'fun' is assumed to return the gradient alog with the objective function.:
    If False, the gradient will be estimated numerically
'''
initial_grad = np.append(np.append(b1, (W1.T).ravel()), np.append(b2, (W2.T).ravel() ) )
Result = op.minimize(fun = nn_costFn_TanH, # can choose nn_costFn_TanH OR nn_costFn_SIGMOID
                                 x0 = initial_grad), 
                                 args = (X, y),
                                 method = 'TNC',
                                 jac = True);
                     
def predict(W1, b1, W2, b2, x):
    # Forward propagation
    a1 = x.copy()
    z2 = a1.dot(W1) + b1
    a2 = np.tanh(z2)
    z3 = a2.dot(W2) + b2
    exp_scores = np.exp(z3)
    a3 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(a3, axis=1)


