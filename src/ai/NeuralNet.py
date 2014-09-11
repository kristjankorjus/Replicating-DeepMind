'''

NeuralNet class creates a Q-learining network by binding together different neural network layers

'''

class NeuralNet:

    def __init__(self, input_shape, filter_shapes, strides, n_hidden, n_out):
        '''
        Initialize a NeuralNet

        @param input_shape: tuple or list of length 4 , (batch size, num input feature maps,
                             image height, image width)
        @param filter_shapes: list of 2 (for each conv layer) * 4 values (number of filters, num input feature maps,
                              filter height,filter width)
        @param strides: list of size 2, stride values for each hidden layer
        @param n_hidden: int, number of neurons in the all-to-all connected hidden layer
        @param n_out: int, number od nudes in output layer
        '''
        pass

    def train(self, minibatch):
        """
        Train function that transforms (state,action,reward,state) into (input, expected_output) for neural net
        and trains the network
        @param minibatch: array of dictionaries, each dictionary contains
        one transition (prestate,action,reward,poststate)
        """
        pass

    def predict_best_action(self, state):
        """
        Predict_best_action returns the action with the highest Q-value
        @param state: 4D array, input (game state) for which we want to know the best action
        """
        pass