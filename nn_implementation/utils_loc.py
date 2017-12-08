# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 15:11:58 2017

@author: Raul Vazquez
"""
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

def plot_decision_boundary(pred_func, X,y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    
    

def nn_costFn_TanH(initial_params, nn_input_dim, nn_hdim, nn_output_dim, X, y,  reg_lambda, *args):
    '''
    Forward propagation to compute the value of the loss function
    Backward propagation to compute the value of the gradient of the loss function 
    '''
     
    # Recover initial form of the NN parameters matrices
    b1 = initial_params[0:nn_hdim]
    cardW1 = nn_input_dim*nn_hdim
    W1 = np.reshape( initial_params[nn_hdim:(nn_hdim + cardW1)], (nn_input_dim, nn_hdim) )
    b2 = initial_params[(nn_hdim + cardW1):(nn_hdim + cardW1 + nn_output_dim)]
    #cardW2 = nn_hdim * nn_output_dim
    W2 = np.reshape( initial_params[(nn_hdim + cardW1 + nn_output_dim):], (nn_hdim, nn_output_dim) )

    #(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
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
    grad = np.append(np.append(db1, dW1.ravel()), np.append(db2, dW2.ravel() ) )
    return (data_loss, grad)


def predict(W1, b1, W2, b2, x):
    # Forward propagation
    a1 = x.copy()
    z2 = a1.dot(W1) + b1
    a2 = np.tanh(z2)
    z3 = a2.dot(W2) + b2
    exp_scores = np.exp(z3)
    a3 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(a3, axis=1) # np.tanh(z3)


def sigmoid(z):
    return (1/(1+np.exp(-z)))

def predict_sigmoid(W1, b1, W2, b2, x):
    ''' using in: 
                nn2.py
    Different than predict, in the fact than nn2.py constructs a single unit output
    hence, there is no need to construct a return np.argmax(a3, axix=1), simply return a3
    '''
    # Forward propagation
    a1 = (x).copy()
    z2 = a1.dot(W1) + b1
    a2 = sigmoid(z2)
    z3 = a2.dot(W2) + b2
    a3 = sigmoid(z3)
    return np.round(a3)

def random_init(nn_input_dim, nn_hdim, nn_output_dim):
    np.random.seed(10)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    #cardW2 = nn_hdim * nn_output_dim
    b2 = np.zeros((1, nn_output_dim))
    return (b1, W1, b2, W2)



def gradient_updates_momentum(cost, params, learning_rate, momentum):
    '''
    Compute updates for gradient descent with momentum
    
    :parameters:
        - cost : theano.tensor.var.TensorVariable
            Theano cost function to minimize
        - params : list of theano.tensor.var.TensorVariable
            Parameters to compute gradient against
        - learning_rate : float
            Gradient descent learning rate
        - momentum : float
            Momentum parameter, should be at least 0 (standard gradient descent) and less than 1
   
    :returns:
        updates : list
            List of updates, one for each parameter
    '''
    # Make sure momentum is a sane value
    assert momentum < 1 and momentum >= 0
    # List of update steps for each parameter
    updates = []
    # Just gradient descent on cost
    for param in params:
        # For each parameter, we'll create a previous_step shared variable.
        # This variable will keep track of the parameter's update step across iterations.
        # We initialize it to 0
        previous_step = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        # Each parameter is updated by taking a step in the direction of the gradient.
        # However, we also "mix in" the previous step according to the given momentum value.
        # Note that we don't need to derive backpropagation to compute updates - just use T.grad!
        step = momentum*previous_step - learning_rate*T.grad(cost, param)
        # Add an update to store the previous step value
        updates.append((previous_step, step))
        # Add an update to apply the gradient descent step to the parameter itself
        updates.append((param, param + step))
    return updates
