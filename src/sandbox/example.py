"""
Minimal example of using theano and neural network
"""

import theano
import theano.tensor as T
import numpy as np
# import theano.printing as tprint


def shared_dataset(data_xy):
    """
    Transform data into theano.shared. This is important for parallelising computations later
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    return shared_x, shared_y


class HiddenLayer:
    """
    Implements hidden layer of 
    """

    def __init__(self, input, n_in, n_nodes):
        self.input = input

        #: Weight matrix (n_in x n_nodes)
        W_values = np.asarray(np.ones((n_in, n_nodes)), dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values, name='W', borrow=True)

        #: Bias term
        b_values = np.zeros((n_nodes,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        #: Output is just the weighted sum of activations
        self.output = T.dot(input, self.W) + self.b


class OutputLayer:
    """
    Implement last layer of the network. Output values of this layer are the results of the computation.
    """

    def __init__(self, input_from_previous_layer, n_in, n_nodes):
        #: Weight matrix (n_in x n_nodes)
        W_values = np.asarray(np.ones((n_in, n_nodes)), dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values, name='W', borrow=True)

        #: Bias term
        b_values = np.zeros((n_nodes,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        #output using linear rectifier
        self.threshold = 1
        lin_output = T.dot(input_from_previous_layer, self.W) + self.b
        above_threshold = lin_output > self.threshold
        self.output = above_threshold * (lin_output - self.threshold)


class MLP:
    """
    Class which implements the classification algorithm (neural network in our case)
    """

    def __init__(self, input_initial, n_in, n_hidden, n_out):
        #: Hidden layer implements summation
        self.hidden_layer = HiddenLayer(input_initial, n_in, n_hidden)

        #: Output layer implements summations and rectifier non-linearity
        self.output_layer = OutputLayer(self.hidden_layer.output, n_hidden, n_out)


def main():
    #: Define datasets
    #train_set = (np.array([[1, 1], [1, 0], [0, 1], [0, 0]]), np.array([1, 0, 0, 0]))
    train_set = (np.array([[[0, 0], [0, 1], [1, 1], [1, 0], [1, 1]], [[0, 0], [0, 1], [1, 1], [1, 0], [1, 1]]]),
                 np.array([0, 0, 1, 0, 1]))
    test_set = (np.array([[0, 0], [1, 0]]), np.array([0, 0]))

    # Transform them to theano.shared
    train_set_x, train_set_y = shared_dataset(train_set)

    test_set_x, test_set_y = shared_dataset(test_set)

    # This is how you can print weird theano stuff
    print train_set_x.eval()
    print train_set_y.eval()

    # Define some structures to store training data and labels
    x = T.matrix('x')
    y = T.ivector('y')
    index = T.lscalar()

    # Define the classification algorithm
    classifier = MLP(input_initial=x, n_in=2, n_hidden=1, n_out=1)

    # we construct an object of type theano.function, which we call test_model
    test_model = theano.function(inputs=[index],
                                 outputs=[classifier.hidden_layer.input, classifier.output_layer.output],
                                 givens={x: train_set_x[index]})

    n_train_points = train_set_x.get_value(borrow=True).shape[0]
    print "nr of training points is ", n_train_points

    for i in range(n_train_points):
        result = test_model(i)
        print "we calculated something: ", result


if __name__ == '__main__':
    main()

