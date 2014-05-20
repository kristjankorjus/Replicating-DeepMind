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


class NeuralNet:

    def __init__(self, input_shape, filter_shapes, strides, n_hidden, n_out):

        x = T.matrix('x')
        y = T.matrix('y')

        self.layer_hidden_conv1 = ConvolutionalLayer(x, filter_shapes[0], input_shape, strides[0])

        second_conv_input_shape=[input_shape[0], filter_shapes[0], filter_shapes[2], filter_shapes[3]]
        self.layer_hidden_conv2 = ConvolutionalLayer(self.layer_hidden_conv1.output, filter_shapes[1],
                                                     image_shape=second_conv_input_shape, stride=2)

        flattened_input=self.layer_hidden_conv2.output.flatten(2)

        self.layer_hidden3 = HiddenLayer(flattened_input, self.layer_hidden_conv2.fan_out, n_hidden)
        self.layer_output = OutputLayer(self.layer_hidden3.output, n_hidden, n_out)
        self.params = self.layer_hidden_conv1.params + self.layer_hidden_conv2.params \
                    + self.layer_hidden3.params + self.layer_output.params

        self.gamma = 0.05

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
        learning_rate = 0.0001
        updates = []
        for param_i, gparam_i in zip(self.params, grads):
            updates.append((param_i, param_i - learning_rate * gparam_i))

        temp1 = T.dmatrix('temp1')
        temp2 = T.dmatrix('temp2')

        self.train_model = theano.function(inputs=[temp1, temp2], outputs=[cost],
            updates=updates,
            givens={
                x: temp1,
                y: temp2})


        self.predict_rewards = theano.function(
            inputs=[temp1, temp2],
            outputs=[self.layer_output.output, cost],
            givens={
                x: temp1,
                y: temp2
            })




    def train(self, minibatch):
        dataset=[]
        states1 = [element['prestate'] for element in minibatch]
        states2 = [element['poststate'] for element in minibatch]
        current_predicted_rewards = self.predict_rewards(states1)
        predicted_future_rewards = self.predict_rewards(states2)
        for i, (prestate, a, r, poststate) in enumerate(minibatch):
            rewards = current_predicted_rewards[i]
            rewards[a] = r+self.gamma*np.max(predicted_future_rewards[i])
            dataset.append([prestate, rewards])
        dataset=np.array(dataset)
        self.train_model(dataset[:, 0], dataset[:, 1])




NeuralNet([32, 4, 84, 84], filter_shapes=[[16, 4, 8, 8], [32, 16, 4, 4]], strides=[4, 2], n_hidden=256, n_out=4)


