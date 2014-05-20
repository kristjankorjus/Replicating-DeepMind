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

    def __init__(self, input, input_shape, filter_shapes, strides, n_hidden, n_out):


        self.layer_hidden_conv1 = ConvolutionalLayer(input, filter_shapes[0], input_shape, strides[0])
        self.layer_hidden_conv2 = ConvolutionalLayer()

        flattened_input=self.layer_hidden_conv2.output.flatten(2)

        self.layer_hidden3 = HiddenLayer(flattened_input, self.layer_hidden_conv2.fan_out, n_hidden)
        self.layer_output = OutputLayer(self.hidden_layer.output, n_hidden, n_out)



        x = T.matrix('x')
        y = T.matrix('y')

        v=T.dmatrix('v')
        yy=T.dvector('yy')

        self.predict_rewards=theano.function(
            inputs=[v,yy],
            outputs=[layer_output.output],
            givens={
                x: v,
                y: yy
            })




    def train(self, mini_batch):
        y = self.predict_rewards(mini_batch[:, 0])
        y2 = np.max( self.predict_rewards(mini_batch[:, 3]), axis=0)




    def get_action(self, images):
        '''
        Image is a 3D matrix: 4 x 84 x 84
        '''
        return random.randint(0, 4)

    def train(self):
        pass
