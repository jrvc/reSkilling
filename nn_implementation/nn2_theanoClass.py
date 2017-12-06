# -*- coding: utf-8 -*-
"""
Layer Class for theano
"""
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

class Layer(object):
    '''Each layer is defined as a class, which stores a weight matrix and
    a bias vector and includes a function for computing the layer's output.
    '''
    def __init__(self, W_init, b_init, activation):
        '''A layer of a neural network, computes s(Wx + b), where
        s is a nonlinearity
        x is the input vector
        W is the parameters matrix
        b the bias unit

        PARAMETERS:
            - W_init : np.ndarray, shape=(n_output, n_input)
                Values to initialize the weight matrix to.
            - b_init : np.ndarray, shape=(n_output,)
                Values to initialize the bias vector
            - activation : theano.tensor.elemwise.Elemwise
                Activation function for layer output
        '''
        # Get dimensions ofthe network:
        n_output, n_input=W_init.shape
        # Make sure b is n_output in size
        assert b_init.shape == (n_output,), 'bias unit has not proper dimensions'
        
        # All parameters should be shared variables, since they're are updated
        # elsewhere when optimizing the network parameters.
        self.W = theano.shared(value=W_init.astype(theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=b_init.reshape(n_output, 1).astype(theano.config.floatX), # force our bias vector b to be a column vector
                               name='b', borrow=True, broadcastable=(False, True)) # By setting broadcastable=(False, True), we are denoting that b can be broadcast (copied) along its second dimension in order to be added to another variable. 
        self.activation = activation
        # We will compute the gradient of the cost of the network with respect to the parameters in this list.
        self.params = [self.W, self.b]
        
        
    def output(self, x):
        '''
        Compute this layer's output given an input
        
        PARAMETERS:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for layer input

        OUTPUT:
            - output : theano.tensor.var.TensorVariable
                Mixed, biased, and activated x
        '''
        lin_output = T.dot(self.W, x) + self.b
        # Output is just linear mix if no activation function
        # Otherwise, apply the activation function
        return (lin_output if self.activation is None else self.activation(lin_output))
    
    
    
    
    
    
class MLP(object):
    '''Multi-layer perceptron class (MLP), computes the composition of a sequence of Layers'''
    def __init__(self, W_init, b_init, activations):
        '''
        PARAMETERS:
            - W_init : list of np.ndarray, len=N
                Values to initialize the weight matrix in each layer to.
                The layer sizes will be inferred from the shape of each matrix in W_init
            - b_init : list of np.ndarray, len=N
                Values to initialize the bias vector in each layer to
            - activations : list of theano.tensor.elemwise.Elemwise, len=N
                Activation function for layer output for each layer
        '''
        # Make sure the input lists are all of the same length
        assert len(W_init) == len(b_init) == len(activations)
        
        # Initialize lists of layers
        self.layers = []
        # Construct the layers
        for W, b, activation in zip(W_init, b_init, activations):
            self.layers.append(Layer(W, b, activation))

        # Combine parameters from all layers
        self.params = []
        for layer in self.layers:
            self.params += layer.params
    
    
    def output(self, x):
        '''
        Compute the MLP's output given an input
        PARAMETERS:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for network input
        OUTPUT:
            - output : theano.tensor.var.TensorVariable
                x passed through the MLP
        '''
        # Recursively compute output
        for layer in self.layers:
            x = layer.output(x)
        return x


    def squared_error(self, x, y):
        '''
        Compute the squared euclidean error of the network output against the "true" output y
        
        PARAMETERS:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for network input
            - y : theano.tensor.var.TensorVariable
                Theano symbolic variable for desired network output
        OUTPUT:
            - error : theano.tensor.var.TensorVariable
                The squared Euclidian distance between the network output and y
        '''
        return T.sum((self.output(x) - y)**2)