# -*- coding: utf-8 -*-
"""
USING THEANO
Build a 3-layer neural network with 1 input layer, 1 hidden layer and 1 output layer. 
    The number of nodes in the input  layer is 2; the dimensionality of our data.
    The number of nodes in the output layer is 2; the number of classes we have.
    The number of nodes in the hidden layer will vary.
Created on Mon Dec  4 12:51:47 2017

@author: Raul Vazquez
"""
#----------------------------------------------------------------
#                  THIS PART IS THE SAME AS nn1.py:
#----------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import scipy.optimize as op
import utils_loc
# Generate a dataset and plot it
np.random.seed(100)
X_train, y_train = datasets.make_moons(200, noise=0.25)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)


m = len(X) # training set size
nn_input_dim = X.shape[1] # input layer dimensionality
nn_output_dim = 2 # output layer dimensionality
''' IMPORTANT PARAMETER: play with nn_hdim to see how the decision boundary changes'''
nn_hdim = 4 # hidden layer dmensionality (3 seems to be the optimal, 4 already overfitts the data)
# Gradient descent parameters
epsilon = 0.01 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength
#---------------------------------------------------------------
#---------------------------------------------------------------



import theano
import theano.tensor as T
 
# Data vectors as tensors
X = T.matrix('X') # matrix of doubles
y = T.lvector('y') # vector of int64
''' We have not assigned any values to X or y.
    All we have done is defined mathematical expressions for them.
    If we want to evaluate an expression we can call its eval method.
    EXAMPLE 1:
        (X * 2).eval({X : [[1,1],[2,2]] })
        evaluates the expression X * 2 for the values in the array X: [[1,1],[2,2]]
    EXAMPLE 2:
        Create a Theano function
        Mat = T.matrix('Mat')
        x_squared = Mat ** 2
        X_call_xsquared = theano.function([Mat], x_squared)
        When called, evaluates the expression defined my x_squared for the values
        in the input given
        X_call_xsquared([[1,1],[2,2]])

'''

# Assign parameters W_1, b_1, W_2, b_2 as shared variables.
# initialize bias uhnits to zero and the weights randomly to break the symmetry of the problem
W1 = theano.shared(np.random.randn(nn_input_dim, nn_hdim), name='W1')
b1 = theano.shared(np.zeros(nn_hdim), name='b1')
W2 = theano.shared(np.random.randn(nn_hdim, nn_output_dim), name='W2')
b2 = theano.shared(np.zeros(nn_output_dim), name='b2')
W1.get_value() # shows the values of W1

''' FORWARD propagation. Same code as in utils_loc.py 
BUT now we are difining expressions, not evaluating. ''' 
a1 = X
z2 = a1.dot(W1) + b1
a2 = T.tanh(z2)
z3 = a2.dot(W2) + b2
a3 = T.nnet.softmax(z3) # recall a3 is the probs of each example to belong to each class
exp_scores = T.exp(z3)
a3bis = exp_scores / T.sum(exp_scores, axis=1, keepdims=True)

# The regularization term 
loss_reg = 1./m * reg_lambda/2 * (T.sum(T.sqr(W1)) + T.sum(T.sqr(W2))) 
# the loss function we want to optimize
data_loss = T.nnet.categorical_crossentropy(a3, y).mean() + loss_reg

# Returns the predicted class for that example/input 
prediction = T.argmax(a3, axis=1)

# Theano functions that can be called from our Python code
fwd_prop = theano.function([X], a3)
fwd_propBIS = theano.function([X], a3bis)
calculate_loss = theano.function([X, y], data_loss)
predict = theano.function([X], prediction)

# Example call: Forward Propagation
# fwd_prop([[4,6]]), fwd_prop([[4,6],[2,3]]) # the argmuent must be of dims m*2 (just 2 cols, since X is like that)

# Calculate the derivatives with Theano
dW2 = T.grad(data_loss, W2)
db2 = T.grad(data_loss, b2)
dW1 = T.grad(data_loss, W1)
db1 = T.grad(data_loss, b1)
''' 
We could here use BACK propagation with the expression definition (as we did for FWDprop):
    y_canonical = T.eye(2)[y]
    delta3 = a3 - y_canonical # recall delta3 = a3 - y, when y defd as a matrix with boolean canonical entries
    delta2 = delta3.dot(W2.T) * (1 - T.power(a2, 2))
    # derivatives
    dW2_noreg = (a2.T).dot(delta3)
    db2 = T.sum(delta3, axis=0, keepdims=True)
    dW1_noreg = np.dot(a1.T, delta2)
    db1 = T.sum(delta2, axis=0)
    # include regularization term
    dW2 = dW2_noreg + reg_lambda * W2
    dW1 = dW1_noreg + reg_lambda * W1
'''

# define a SIMPLE GRADIENT DESCENT in THEANO
gradient_step = theano.function( [X, y],
    updates=((W2, W2 - epsilon * dW2),
             (W1, W1 - epsilon * dW1),
             (b2, b2 - epsilon * db2),
             (b1, b1 - epsilon * db1)))


def train_model(num_iters=20000, print_loss=False):
    '''
     This function learns parameters for the neural network and returns the model.
    INPUT: 
        - num_passes: Number of passes through the training data for gradient descent
        - print_loss: If True, print the loss every 1000 iterations
        '''
    # Re-Initialize the parameters to random values. We need to learn these.
    # (Needed in case we call this function multiple times)
    np.random.seed(0)
    W1.set_value(np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim))
    b1.set_value(np.zeros(nn_hdim))
    W2.set_value(np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim))
    b2.set_value(np.zeros(nn_output_dim))
    
    # Gradient descent. For each batch...
    for i in range(0, num_iters):
        # This will update our parameters W2, b2, W1 and b1!
        gradient_step(X_train, y_train)
        
        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
            print ("Loss after iteration %i: %f" %(i, calculate_loss(X_train, y_train)) )


# Build a model with a 3-dimensional hidden layer
train_model(print_loss=True)

# Plot the decision boundary
utils_loc.plot_decision_boundary(lambda x: predict(x))
plt.title("Decision Boundary for hidden layer size "+ str( nn_hdim))





'''still working on this part:
    trying to answer:
    ????
    Can one use Theano functions in op.minimize()????
    
W1.set_value(np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim))
b1.set_value(np.zeros(nn_hdim))
W2.set_value(np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim))
b2.set_value(np.zeros(nn_output_dim))
initial_grad = np.append(np.append(b1, (W1).ravel()), np.append(b2, (W2).ravel() ) )

call_gradient = theano.function( [X, y],  ((b1,db1),(W1, dW1),(b2, db2),(W2, dW2) ) ) 


def caller(x):
    grad = call_gradient
    return (data_loss, grad)
    return updates(x, nn_input_dim, nn_hdim, nn_output_dim, X, y, reg_lambda)

result = op.minimize(fun = caller, x0 = initial_grad, jac = True)
'''