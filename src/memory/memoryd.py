'''

Memory stores game data and provides means work with it

'''

import numpy as np
import random

class MemoryD:

    #: N x 84 x 84 matrix, where N is the number of game steps we want to keep
    screens = None
    
    #: list of size N, stores actions agent took
    actions = None
    
    #: list of size N, stores rewards agent received upon doing an action
    rewards = None

    #: list of size N, stores game internal time
    time    = None

    #: global index counter
    count   = 0

    def __init__(self, N):
        '''
        Initialize memory structure 
        @param N: the number of game steps we need to store
        '''
        self.screens = np.zeros((N, 84, 84), dtype=np.uint8)
        self.actions = np.zeros((N,), dtype=np.uint8)
        self.rewards = np.zeros((N,), dtype=np.uint8)
        self.time    = np.zeros((N,), dtype=np.uint32)
        self.count   = 0

    def add_first(self, next_screen):
        '''
        When a new game start we add initial game screen to the memory
        @param screen: 84 x 84 np.uint8 matrix
        '''
        self.screen[self.count + 1] = next_screen
        self.time[self.count + 1] = 0
        self.count += 1

    def add(self, action, reward, next_screen):
        '''
        During the game we add few thing to memory
        @param action: the action agent decided to do
        @param reward: the reward agent received for his action
        @param next_screen: next screen of the game
        '''
        self.actions[self.count] = action
        self.rewards[self.count] = reward
        self.time[self.count + 1] = self.time[self.count] + 1
        self.screen[self.count + 1]  = next_screen
        self.count += 1

    def add_last(self):
        '''
        When the game ends we fill memory for the current screen with corresponding
        values. It is useful to think that at this time moment agent is looking at
        "Game Over" screen
        '''
        self.actions[self.count] = None
        self.rewards[self.count] = None

    def get_minibatch(self, size):
        '''
        Take n Transitions from the memory.
        One transition includes (state, action, reward, state)
        '''

        transitions = []

        #: Pick random n indices
        indices = [random.randint(0, self.count - 1) for i in range(size)]
        for i in indices:
            if self.actions[i] is not None:
                transitions.append({'prestate': self.get_state(i),
                                    'action': self.actions[i],
                                    'reward': self.rewards[i],
                                    'poststate': self.get_state(i + 1)})

        return transitions


    def get_state(self, index):
        '''
        Extract one state (4 images) given last image position
        @param index: global location of the 4th image in the memory
        '''

        #: We always need 4 images to compose one state. In the beginning of the
        #  game (at time moments 0, 1, 2) we do not have enough images in the memory
        #  for this particular game. So we came up with an uglyhack: duplicate the
        #  first available image as many times as needed to fill missing ones.
        pad_screens = 3 - self.time[index]
        if pad_screens > 0:

            state = []

            #: Pad missing images with the first image
            for p in range(pad_screens):
                state.append(self.screens[index - 3 + pad_screens])

            #: Fill the rest of the images as they are
            for p in range(pad_screens, 4):
                state.append(self.screens[index - (4 - p - 1)])

        else:
            state = self.screens[index - 3 : index + 1]

        return state

    def get_last_state(self):
        '''
        Get last 4 images from the memory. Those images will go as an input for
        the neural network
        '''
        return self.get_state(self.count)
