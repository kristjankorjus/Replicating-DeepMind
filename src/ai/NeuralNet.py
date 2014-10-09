'''

NeuralNet class creates a Q-learining network by binding together different neural network layers

'''

from convnet import *
import numpy as np

class DeepmindDataProvider:
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        pass

    def get_data_dims(self, idx=0):
		# TODO: these numbers shouldn't be hardcoded
        if idx == 0:
            return 4*84*84
        if idx == 1:
            return 4
        return 1

class NeuralNet(ConvNet):

    def __init__(self, output_layer_name = 'layer4', discount_factor = 0.9):
        '''
        Initialize a NeuralNet

        @param output_layer_name: name of the output (actions) layer
        '''
        # Initialise ConvNet, including self.libmodel
        op = NeuralNet.get_options_parser()
        print op.options.keys()
        op, load_dic = IGPUModel.parse_options(op)
        print op.options.keys()
        ConvNet.__init__(self, op, load_dic)

		# Remember parameters and some useful variables
        self.discount = discount_factor
        self.output_layer_name = output_layer_name
        self.num_outputs = self.layers[output_layer_name]['outputs']

    def train(self, minibatch):
        """
        Train function that transforms (state,action,reward,state) into (input, expected_output) for neural net
        and trains the network
        @param minibatch: array of dictionaries, each dictionary contains
        one transition (prestate,action,reward,poststate)
        """
        states = minibatch[0]
        actions = minibatch[1]
        rewards = minibatch[2]
        next_states = minibatch[3]
        nextrewards = np.max(self.predict(next_states), axis=0)
        scores = self.predict(states)
        # we have to update the Q-vals for the actions we actually performed
        for i,action in enumerate(actions):
            scores[i][action]= rewards[i] + self.discount * nextrewards[i]
        
        self.libmodel.startBatch([states, scores], 1, false) # second parameter is 'progress', third parameter means 'only test, don't train'
        cost = self.libmodel.finishBatch()
        return cost

    def predict(self, states):
        """
        Predict returns neural network output layer activations for input
        @param input: input data for neural network
        """
        batch_size = np.shape(states)[0]
        scores = np.zeros((batch_size, self.num_outputs), dtype=np.single)
        self.libmodel.startFeatureWriter(states, [scores], [self.output_layer_name])
        self.libmodel.finishBatch()
        # now activations of output layer should be in 'scores'
        return scores

    def predict_best_action(self, state):
		# predict() expects input as a matrix
        states = np.reshape(last_state, (1,len(last_state)))
        print "shape of predict", np.shape(self.predict(states))
		# return action with maximum score
        return np.argmax(self.predict(states)[0])

    #remove options we do not need
    @classmethod
    def get_options_parser(cls):
        op = ConvNet.get_options_parser()
        #op.delete_option("train_batch_range")
        #op.delete_option("test_batch_range")
        #op.delete_option("dp_type")
        #op.delete_option("data_path")
        op.options["train_batch_range"].default="0"
        op.options["test_batch_range"].default="0"
        op.options["dp_type"].default="image"
        op.options["data_path"].default="/storage/hpc_kristjan/cuda-convnet4" # TODO: remove this

        DataProvider.register_data_provider('deepmind', 'DeepMind data provider', DeepmindDataProvider)

        return op
