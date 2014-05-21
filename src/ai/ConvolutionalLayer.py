'''

Creates one convolutional layer

'''


import theano
import numpy as np
from theano.tensor.nnet import conv


class ConvolutionalLayer(object):

    def __init__(self, input_images, filter_shape, image_shape, stride):
        '''
        Initialize a ConvolutionalLayer
        @param input_images: theano.tensor.dtensor4 , symbolic image tensor, of shape image_shape

        @param filter_shape: tuple or list of length 4 , (number of filters, num input feature maps,
                              filter height,filter width)

        @param image_shape: tuple or list of length 4 , (batch size, num input feature maps,
                             image height, image width)

        @param stride: the number of values to skip when convolving over input feature maps
        '''

        #: we want to be sure that these two values are the same
        assert image_shape[1] == filter_shape[1]


        self.input = input_images

        #: there are "num input feature maps * filter height * filter width"
        #  inputs to each hidden unit
        self.fan_in = np.prod(filter_shape[1:])

        # number of nodes in our layer (size of output)
        self.feature_map_size=1+(image_shape[2]-filter_shape[2])/stride
        self.fan_out = (filter_shape[0] * self.feature_map_size * self.feature_map_size)


        #: we need to define the interval at which we initialize the weights. We use formula from example
        W_bound = np.sqrt(6. / (self.fan_in + self.fan_out))

        #: initialize weights with random weights
        self.W = theano.shared(np.asarray(np.random.uniform(high=W_bound, low=-W_bound, size=filter_shape), dtype=theano.config.floatX),
                               borrow=True)

        #: the bias is a 1D vector -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        #: convolve input feature maps with filters
        convolution_output = conv.conv2d(input=input_images, filters=self.W, filter_shape=filter_shape,
                                         image_shape=image_shape, subsample=(stride , stride))

        #: Since the bias is a vector (1D array), we need first
        #  reshape it to a tensor of shape (1,n_filters,1,1) using dimshuffle
        self.threshold = 0
        activation = convolution_output + self.b.dimshuffle('x', 0, 'x', 'x')
        above_threshold = activation > self.threshold
        self.output = above_threshold * (activation - self.threshold)

        #: store parameters of this layer
        self.params = [self.W, self.b]
