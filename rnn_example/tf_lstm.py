# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn as tflearn
from tensorflow.contrib import layers as tflayers


def lstm_model(num_units, rnn_layers, dense_layers=None, learn_rate=0.1, optimizer='Adagrad'):
    """
    Creates a deep NN model with:
        + stacked lstm cells
        + (optional) dense layers
    i.e, creates a Long Short Term Memory (lstm) model.
    INPUT:
        - num_units: the size of the cells (number of timesteps to take into account).
        - rnn_layers: list of int or dict
                     * list of int: the steps used to instantiate the `BasicLSTMCell` cell
                     * list of dict: [{num_units: int, keep_prob: int}, ...] 
                                     the list includes one dictionary for every lstm layer
        - dense_layers: list of int. 
                        Each entry of the list indicates the number of nodes for each layer
    OUTPUT:
        the model definition
    """

    def lstm_cells(layers):
        '''
        Defines the LSTM layers.
        INPUT:
            - layers: same as rnn_layers (list of int or list of dict)
        '''
        # CASE := layers is a list of dict: for each dict create a BasicLSTMCell layer with the specifications in it
        if isinstance(layers[0], dict):
            return [tf.contrib.rnn.DropoutWrapper( # Operator adding dropout to inputs and outputs of the given cell.
                    tf.contrib.rnn.BasicLSTMCell(layer['num_units'], state_is_tuple=True),
                    layer['keep_prob'] ) 
                if layer.get('keep_prob') else 
                    tf.contrib.rnn.BasicLSTMCell(layer['num_units'], state_is_tuple=True) 
                for layer in layers] 
        # CASE := layers is a list of int: creat a BasicLSTMCell layer with the steps
        return [tf.contrib.rnn.BasicLSTMCell(steps, state_is_tuple=True) for steps in layers]


    def dnn_layers(input_layers, layers):
        '''
        Defines the optional dense layers
         *note: A dense layer is a kind of hidden layer where every node is connected
                to every other node in the next layer
        INPUT:
            - input_layers:
            - layers: same as dense_layers
        '''
        if layers and isinstance(layers, dict):
            return tflayers.stack(input_layers, tflayers.fully_connected,
                                  layers['layers'],
                                  activation=layers.get('activation'),
                                  dropout=layers.get('dropout'))
        elif layers:
            return tflayers.stack(input_layers, tflayers.fully_connected, layers)
        else:
            return input_layers

    def _lstm_model(X, y):
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells(rnn_layers), state_is_tuple=True) # MultiRNNCell: RNN cell composed sequentially of multiple simple cells
        x_ = tf.unstack(X, axis=1, num=num_units) # Unpacks num_units tensors from X by chipping it along the axis dimension. 
        output, layers = tf.contrib.rnn.static_rnn(stacked_lstm, x_, dtype=dtypes.float32) # Creates a RNN specified by RNNCell
        output = dnn_layers(output[-1], dense_layers) # output has now passed the LSTM and the dense layers
        prediction, loss = tflearn.models.linear_regression(output, y)
        train_op = tf.contrib.layers.optimize_loss( loss, tf.contrib.framework.get_global_step(), optimizer=optimizer, learning_rate=learn_rate)
        return prediction, loss, train_op

    return _lstm_model