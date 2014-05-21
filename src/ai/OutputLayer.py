'''
    Implement last layer of the network. Output values of this layer are the results of the computation.
'''

import theano
import numpy as np
import theano.tensor as T


class OutputLayer:

    def __init__(self, input_from_previous_layer, n_in, n_nodes):
        '''
        Initialize an output layer. The only difference with hidden layer is that no linear rectifier is used
        @param input_from_previous_layer: theano.tensor.dmatrix of shape (batch_size,n_in), represents inputs from previous layer
        @param n_in: int, number of inputs to layer
        @param n_nodes: int, number of nodes in the layer. Also the size of output
        '''

        self.input=input_from_previous_layer

        #: we need to limit the weight sizes because we might have many inputs to each node
        W_bound = -np.sqrt(6. / (n_in + n_nodes))

        #: Weight matrix (n_in x n_nodes)
        W_values = np.asarray(np.random.uniform(high=W_bound, low=-W_bound, size=(n_in, n_nodes)), dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values, name='W', borrow=True)

        #: Bias term
        b_values = np.zeros((n_nodes,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        #: output layer does not use linear rectifier
        lin_output = T.dot(input_from_previous_layer, self.W) + self.b
        self.output = lin_output

        #: all the variables that can change during learning
        self.params = [self.W, self.b]


    def errors(self, y):
        ''' return the error made in predicting the output value
        @param y: theano.tensor.TensorType, corresponds to a vector that gives for each node
        the value we actually wished to obtain
        '''

        #: check if y has same dimension as output
        if y.ndim != self.output.ndim:
            raise TypeError('y should have the same shape as self.output', ('y', y.type, 'output', self.output.type))

        return T.mean(np.abs(self.output-y))
