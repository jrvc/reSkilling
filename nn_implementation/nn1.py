# -*- coding: utf-8 -*-
"""
FROM SCRATCH
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

@author: Raul Vazquez
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import scipy.optimize as op
import utils_loc

# Generate a dataset and plot it
np.random.seed(100)
X, y = datasets.make_moons(200, noise=0.25)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)


m = len(X) # training set size
nn_input_dim = X.shape[1] # input layer dimensionality
nn_output_dim = 2 # output layer dimensionality
''' IMPORTANT PARAMETER: play with nn_hdim to see how the decision boundary changes'''
nn_hdim = 4 # hidden layer dmensionality (3 seems to be the optimal, 4 already overfitts the data)
 
# Gradient descent parameters
epsilon = 0.01 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength

# Initialize the parameters to random values. We need to learn these.
np.random.seed(10)
b1, W1, b2, W2 = utils_loc.random_init(nn_input_dim, nn_hdim, nn_output_dim)
cardW1 = nn_input_dim * nn_hdim
#cardW2 = nn_hdim * nn_output_dim

# generate an unrolled vector of parameters
initial_grad = np.append(np.append(b1, (W1).ravel()), np.append(b2, (W2).ravel() ) )

# minimize the desired function
'''
if 'jac' is a boolean and is True, 'fun' is assumed to return the gradient alog with the objective function.:
    If False, the gradient will be estimated numerically
'''
def caller(x):
    return utils_loc.nn_costFn_TanH(x, nn_input_dim, nn_hdim, nn_output_dim, X, y, reg_lambda)

result = op.minimize(fun = caller, x0 = initial_grad, jac = True)
# for fun one can choose nn_costFn_TanH OR nn_costFn_SIGMOID    
result = op.minimize(fun = utils_loc.nn_costFn_TanH, x0 = initial_grad,
                     args = (nn_input_dim, nn_hdim, nn_output_dim, X, y,  reg_lambda), 
                     jac = True)

new_b1 = result.x[0:nn_hdim]
new_W1 = np.reshape( result.x[nn_hdim:(nn_hdim + cardW1)], (nn_input_dim, nn_hdim) )
new_b2 = result.x[(nn_hdim + cardW1):(nn_hdim + cardW1 + nn_output_dim)]
new_W2 = np.reshape( result.x[(nn_hdim + cardW1 + nn_output_dim):], (nn_hdim, nn_output_dim) )
    

utils_loc.plot_decision_boundary( (lambda x: utils_loc.predict(W1, b1, W2, b2, x)), X, y )
plt.title("Decision Boundary of initial Parameters")
                     
utils_loc.plot_decision_boundary( (lambda x: utils_loc.predict(new_W1, new_b1, new_W2, new_b2, x) ), X,y)
plt.title("Decision Boundary for hidden layer size "+ str( nn_hdim))


