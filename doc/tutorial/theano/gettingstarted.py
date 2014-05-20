'''
Getting Started with Theano and Deep Learning

http://deeplearning.net/tutorial/gettingstarted.html
'''

import cPickle, gzip, numpy
import theano
import theano.tensor as T

# Load the dataset
f = gzip.open('mnist.pkl.gz')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

def shared_dataset(data_xy):
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
    return shared_x, T.cast(shared_y, 'int32')

test_set_x, test_set_y = shared_dataset(test_set)
valid_set_x, valid_set_y = shared_dataset(valid_set)
train_set_x, train_set_y = shared_dataset(train_set)

# Minibatch
batch_size = 500
data = train_set_x[2 * 500: 3 * 500]
label = train_set_y[2 * 500: 3 * 500]

print data[0].value