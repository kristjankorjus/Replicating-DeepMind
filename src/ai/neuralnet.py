'''

NeuralNet class implements neural network and all related stuff
Probably will be split into several classes

'''

import random

class NeuralNet:

    def get_action(images):
        '''
        Image is a 3D matrix: 4 x 84 x 84
        '''
        return random.randint(0, 4)