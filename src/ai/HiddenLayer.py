import theano
import numpy as np
import theano.tensor as T

class HiddenLayer:
    """
    Implements hidden layer of
    """

    def __init__(self, input, n_in, n_nodes):
        self.input = input

        W_bound = np.sqrt(6. /(n_in+n_nodes))
        #: Weight matrix (n_in x n_nodes)
        W_values = np.asarray(np.random.uniform(high=W_bound, low=-W_bound, size=(n_in, n_nodes)), dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values, name='W', borrow=True)

        #: Bias term
        b_values = np.zeros((n_nodes,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        self.threshold=0
        #: Output is rectified
        dot_product = T.dot(input, self.W) + self.b
        above_threshold = dot_product>self.threshold
        self.output = above_threshold * (dot_product-self.threshold)

        #all the variables that can change during learning
        self.params = [self.W, self.b]
