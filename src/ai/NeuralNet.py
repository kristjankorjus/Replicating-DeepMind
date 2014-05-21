'''

NeuralNet class creates a Q-learining network by binding together different neural network layers

'''

from ConvolutionalLayer import *
from HiddenLayer import *
from OutputLayer import *

theano.config.openmp = True


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

        #create theano variables corresponding to input_batch (x) and output of the network (y)
        x = T.dtensor4('x')
        y = T.dmatrix('y')

        #first hidden layer is convolutional:
        self.layer_hidden_conv1 = ConvolutionalLayer(x, filter_shapes[0], input_shape, strides[0])

        #second convolutional hidden layer: the size of input depends on the size of output from first layer
        #it is defined as (num_batches, num_input_feature_maps, height_of_input_maps, width_of_input_maps)
        second_conv_input_shape = [input_shape[0], filter_shapes[0][0], self.layer_hidden_conv1.feature_map_size,
                                   self.layer_hidden_conv1.feature_map_size]
        self.layer_hidden_conv2 = ConvolutionalLayer(self.layer_hidden_conv1.output, filter_shapes[1],
                                                     image_shape=second_conv_input_shape, stride=2)

        #output from convolutional layer is 4D, but normal hidden layer expects 2D. Because of all to all connections
        # 3rd hidden layer does not care from which feature map or from which position the input comes from
        flattened_input = self.layer_hidden_conv2.output.flatten(2)

        #create third hidden layer
        self.layer_hidden3 = HiddenLayer(flattened_input, self.layer_hidden_conv2.fan_out, n_hidden)

        #create output layer
        self.layer_output = OutputLayer(self.layer_hidden3.output, n_hidden, n_out)

        #define the ensemble of parameters of the whole network
        self.params = self.layer_hidden_conv1.params + self.layer_hidden_conv2.params \
            + self.layer_hidden3.params + self.layer_output.params

        #discount factor
        self.gamma = 0.95

        #: define regularization terms, for some reason we only take in count the weights, not biases)
        #  linear regularization term, useful for having many weights zero
        self.l1 = abs(self.layer_hidden_conv1.W).sum() \
            + abs(self.layer_hidden_conv2.W).sum() \
            + abs(self.layer_hidden3.W).sum() \
            + abs(self.layer_output.W).sum()

        #: square regularization term, useful for forcing small weights
        self.l2_sqr = (self.layer_hidden_conv1.W ** 2).sum() \
            + (self.layer_hidden_conv2.W ** 2).sum() \
            + (self.layer_hidden3.W ** 2).sum() \
            + (self.layer_output.W ** 2).sum()

        #: define the cost function
        cost = 0.0 * self.l1 + 0.0 * self.l2_sqr + self.layer_output.errors(y)

        #: define gradient calculation
        grads = T.grad(cost, self.params)

        #: Define how much we need to change the parameter values
        learning_rate = 0.01
        updates = []
        for param_i, gparam_i in zip(self.params, grads):
            updates.append((param_i, param_i - learning_rate * gparam_i))

        #: we need another set of theano variables (other than x and y) to use in train and predict functions
        temp_x = T.dtensor4('temp1')
        temp_y = T.dmatrix('temp2')

        #: define the training operation as applying the updates calculated given temp_x and temp_y
        self.train_model = theano.function(inputs=[temp_x, temp_y],
                                           outputs=[cost],
                                           updates=updates,
                                           givens={
                                               x: temp_x,
                                               y: temp_y})

        self.predict_rewards = theano.function(
            inputs=[temp_x],
            outputs=[self.layer_output.output],
            givens={
                x: temp_x
            })

        self.predict_rewards_and_cost = theano.function(
            inputs=[temp_x, temp_y],
            outputs=[self.layer_output.output, cost],
            givens={
                x: temp_x,
                y: temp_y
            })


    def train(self, minibatch):
        """
        Train function that transforms (state,action,reward,state) into (input, expected_output) for neural net
        and trains the network
        @param minibatch: array of dictionaries, each dictionary contains
        one transition (prestate,action,reward,poststate)
        """

        #: array of size (batch_size,4), that will be filled with our best estimation on what Q-values should be like.
        #  Corresponds to the "expected_output" or "y" of the neural net
        expected_qs = []

        initial_states = [element['prestate'] for element in minibatch]
        states2 = [element['poststate'] for element in minibatch]

        #: our network's current estimation of Q-s for each possible action
        current_predicted_rewards = self.predict_rewards(initial_states)[0]

        #: the Q-s of the state we actually went to (by choosing an action), we actually only want the maximum of them
        predicted_future_rewards = self.predict_rewards(states2)[0]

        #: we have a new, better estimation for the Q-val of the action we chose, it is the sum of the reward
        #  received on transition and the maximum of future rewards. Q-s for other actions remain the same.
        for i, transition in enumerate(minibatch):
            rewards = current_predicted_rewards[i]
            rewards[transition['action']] = transition['reward'] + self.gamma * np.max(predicted_future_rewards[i])
            expected_qs.append(rewards)

        #: knowing what expected_qs looks like, we can train the model
        self.train_model(initial_states, expected_qs)


    def predict_best_action(self, state):
        """
        Predict_best_action returns the action with the highest Q-value
        @param state: 4D array, input (game state) for which we want to know the best action
        """
        predicted_values_for_actions = self.predict_rewards(state)[0]
        return np.argmax(predicted_values_for_actions)