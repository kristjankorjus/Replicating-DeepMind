'''

NeuralNet class implements neural network and all related stuff
Probably will be split into several classes

'''

import random
import theano.tensor as T
import theano
from ConvolutionalLayer import *
from HiddenLayer import *
from OutputLayer import *

theano.config.openmp = True

class NeuralNet:

    def __init__(self, input_shape, filter_shapes, strides, n_hidden, n_out):

        x = T.dtensor4('x')
        y = T.dmatrix('y')

        self.layer_hidden_conv1 = ConvolutionalLayer(x, filter_shapes[0], input_shape, strides[0])


        second_conv_input_shape = [input_shape[0], filter_shapes[0][0], self.layer_hidden_conv1.feature_map_size, self.layer_hidden_conv1.feature_map_size]
        self.layer_hidden_conv2 = ConvolutionalLayer(self.layer_hidden_conv1.output, filter_shapes[1],
                                                     image_shape=second_conv_input_shape, stride=2)

        flattened_input=self.layer_hidden_conv2.output.flatten(2)

        self.layer_hidden3 = HiddenLayer(flattened_input, self.layer_hidden_conv2.fan_out, n_hidden)
        self.layer_output = OutputLayer(self.layer_hidden3.output, n_hidden, n_out)
        self.params = self.layer_hidden_conv1.params + self.layer_hidden_conv2.params \
                    + self.layer_hidden3.params + self.layer_output.params

        self.gamma = 0.95

        self.L1 = abs(self.layer_hidden_conv1.W).sum() \
                + abs(self.layer_hidden_conv2.W).sum() \
                + abs(self.layer_hidden3.W).sum()  \
                + abs(self.layer_output.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.layer_hidden_conv1.W ** 2).sum() \
                    + (self.layer_hidden_conv2.W ** 2).sum() \
                    + (self.layer_hidden3.W ** 2).sum() \
                    + (self.layer_output.W ** 2).sum()



        cost = 0.0*self.L1 + 0.0*self.L2_sqr + self.layer_output.errors(y)

        grads = T.grad(cost, self.params)

         # Define how much we need to change the parameter values
        learning_rate = 0.01
        updates = []
        for param_i, gparam_i in zip(self.params, grads):
            updates.append((param_i, param_i - learning_rate * gparam_i))

        temp1 = T.dtensor4('temp1')
        temp2 = T.dmatrix('temp2')


        self.train_model = theano.function(inputs=[temp1, temp2], outputs=[cost],
            updates=updates,
            givens={
                x: temp1,
                y: temp2})



        self.predict_rewards = theano.function(
            inputs=[temp1],
            outputs=[self.layer_output.output],
            givens={
                x: temp1
            })

        self.predict_rewards_and_cost = theano.function(
            inputs=[temp1, temp2],
            outputs=[self.layer_output.output, cost],
            givens={
                x: temp1,
                y: temp2
            })



    def train(self, minibatch):
        states = []
        expected_Qs = []
        states1 = [element['prestate'] for element in minibatch]
        states2 = [element['poststate'] for element in minibatch]
        current_predicted_rewards = self.predict_rewards(states1)[0]

        predicted_future_rewards = self.predict_rewards(states2)[0]
        for i, transition in enumerate(minibatch):
            rewards = current_predicted_rewards[i]
            rewards[transition['action']] = transition['reward'] + self.gamma*np.max(predicted_future_rewards[i])
            states.append(transition['prestate'])
            expected_Qs.append(rewards)

        self.train_model(states, expected_Qs)

    def predict_best_action(self, state):
        predicted_values_for_actions = self.predict_rewards(state)[0]
        return np.argmax(predicted_values_for_actions)




"""
deepMind = NeuralNet([32, 4, 84, 84], filter_shapes=[[16, 4, 8, 8], [32, 16, 4, 4]], strides=[4, 2], n_hidden=256, n_out=4)

#generateRandomData
Xs = []
Ys = []
for i in range(32):
    images=[]
    for j in range(4):
        images.append(chessboard.make_chessboard(8))
    Xs.append(images)
    Ys.append([1, 0, 0, 0])

out, remaining_error = deepMind.predict_rewards_and_cost(Xs, Ys)
print out
print remaining_error

#minibatch for training does not contain "y"
mb = [{'prestate': [chessboard.make_chessboard(8)]*4, 'action':0, 'reward':1, 'poststate':[chessboard.make_chessboard(8)]*4}]*32

for i in range(100):
    deepMind.train(mb)
    q,c= deepMind.predict_rewards_and_cost(Xs, Ys)
    print q[0], c


out, remaining_error = deepMind.predict_rewards_and_cost(Xs, Ys)
print out[0]
print remaining_error
"""