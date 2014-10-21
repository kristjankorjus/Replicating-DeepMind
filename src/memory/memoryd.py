"""

Memory stores game data and provides means work with it

"""

#TODO solve the issue of when we reach the limit of 1M samples, need to change add functions and solve the question of what to do on the border of new and old samples

import numpy as np
import random


class MemoryD:

    #: size of the memory                                                                                                                                                                                   
    size = None

    #: N x 84 x 84 matrix, where N is the number of game steps we want to keep
    screens = None
    
    #: list of size N, stores actions agent took
    actions = None
    
    #: list of size N, stores rewards agent received upon doing an action
    rewards = None

    #: list of size N, stores game internal time
    time = None

    #: global index counter
    count = None


    def __init__(self, n):
        """
        Initialize memory structure 
        :type self: object
        @param n: the number of game steps we need to store
        """
        self.size = n
        self.screens = np.empty((n, 84, 84), dtype=np.float32)
        self.actions = np.empty((n,), dtype=np.uint8)
        self.rewards = np.empty((n,), dtype=np.float32)
        self.time = np.empty((n,), dtype=np.uint32)
        self.count = -1

    def add_first(self, next_screen):
        """
        When a new game start we add initial game screen to the memory
        @param next_screen: 84 x 84 np.uint8 matrix
        """
        self.screens[(self.count + 1) % self.size] = next_screen
        self.time[(self.count + 1) % self.size] = 0
        self.count += 1

    def add(self, action, reward, next_screen):
        """
        During the game we add few thing to memory
        @param action: the action agent decided to do
        @param reward: the reward agent received for his action
        @param next_screen: next screen of the game
        """
        self.actions[(self.count) % self.size] = action
        self.rewards[(self.count) % self.size] = reward
        self.time[(self.count + 1) % self.size] = self.time[(self.count) % self.size] + 1
        self.screens[(self.count + 1) % self.size] = next_screen
        self.count += 1

    def add_last(self):
        """
        When the game ends we fill memory for the current screen with corresponding
        values. It is useful to think that at this time moment agent is looking at
        "Game Over" screen
        """
        self.actions[(self.count) % self.size] = 100
        self.rewards[(self.count) % self.size] = 100

    def get_minibatch(self, size):
        """
        Take n Transitions from the memory.
        One transition includes (state, action, reward, state)
        Returns ndarray with 4 lists inside (at Tambet's request)
        @param size: size of the minibatch (in our case it should be 32)
        """
        #print "get_minibatch()"
        prestates = np.empty((84 * 84 * 4, size), dtype = np.float32)
        actions = np.empty((size), dtype = np.float32)
        rewards = np.empty((size), dtype = np.float32)
        poststates = np.empty((84 * 84 * 4, size), dtype = np.float32)
        #: Pick random n indices and save dictionary if not terminal state
        j = 0
        while j < size:
            i = random.randint(0, np.min([self.count, self.size]) - 1)
            # we don't want to take frames that are just after our rewrite point
            while i > (self.count % self.size) and (i-(self.count % self.size)) < 4:
                i = random.randint(0, np.min([self.count, self.size]) - 1)

            if self.actions[i] != 100: # if not endstate
                prestates[:,j] = self.get_state(i)
                actions[j] = self.actions[i]
                rewards[j] = self.rewards[i]
                poststates[:,j] = self.get_state(i + 1)
                j += 1

        return [prestates, actions, rewards, poststates]

    def get_state(self, index):
        """
        Extract one state (4 images) given last image position
        @param index: global location of the 4th image in the memory
        """

        #print "get_state(", index, ")"

        #: We always need 4 images to compose one state. In the beginning of the
        #  game (at time moments 0, 1, 2) we do not have enough images in the memory
        #  for this particular game. So we came up with an ugly hack: duplicate the
        #  first available image as many times as needed to fill missing ones.

        index = index % self.size
        #print index, self.size, self.count, self.time[index]
        #print "###################"
        pad_screens = 3 - self.time[index]
        if pad_screens > 0:

            state = np.empty((4, 84, 84), dtype = np.float32)

            #: Pad missing images with the first image
            for p in range(pad_screens):
                state[p] = self.screens[index - 3 + pad_screens]

            #: Fill the rest of the images as they are
            for p in range(pad_screens, 4):
                state[p] = self.screens[index - 3 + p]

        else:
            ind_start_window = index - 3
            ind_end_window = index + 1
            if ind_start_window < 0:
                state = np.vstack([self.screens[ind_start_window:], self.screens[:ind_end_window]])
            else:
                state = self.screens[index - 3:index + 1]

	# neural network expects flat input and np.ravel does just that
        #print "returning state: ", np.ravel(state).shape
        return np.ravel(state)

    def get_last_state(self):
        """
        Get last 4 images from the memory. Those images will go as an input for
        the neural network
        """

        #print "get_last_state()"
        return self.get_state(self.count)
