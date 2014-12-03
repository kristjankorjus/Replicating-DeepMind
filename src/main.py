"""

This is the main class where all thing are put together

"""

from ai.NeuralNet import NeuralNet
from memory.memoryd import MemoryD
from ale.ale import ALE
import random
import numpy as np
import time
import sys
from os import linesep as NL

class Main:
    # How many transitions to keep in memory?
    memory_size = 1000000

    # Size of the mini-batch, 32 was given in the paper
    minibatch_size = 32

    # Number of possible actions in a given game, 4 for "Breakout"
    number_of_actions = 4

    # Size of one frame
    frame_size = 84*84

    # Size of one state is four 84x84 screens
    state_size = 4 * frame_size

    # Discount factor for future rewards
    discount_factor = 0.9

    # Exploration rate annealing speed
    epsilon_frames = 1000000.0

    # Epsilon during testing
    test_epsilon = 0.05

    # Total frames played, only incremented during training
    total_frames_trained = 0

    # Number of random states to use for calculating Q-values
    nr_random_states = 100

    # Random states that we use to calculate Q-values
    random_states = None

    # Memory itself
    memory = None

    # Neural net
    nnet = None

    # Communication with ALE
    ale = None

    def __init__(self):
        self.memory = MemoryD(self.memory_size)
        self.ale = ALE(self.memory)
        self.nnet = NeuralNet(self.state_size, self.number_of_actions, "ai/deepmind-layers.cfg", "ai/deepmind-params.cfg", "layer4")

    def compute_epsilon(self, frames_played):
        """
        From the paper: "The behavior policy during training was epsilon-greedy
        with annealed linearly from 1 to 0.1 over the first million frames, and fixed at 0.1 thereafter."
        @param frames_played: How far are we with our learning?
        """
        return max(1.0 - frames_played / self.epsilon_frames, 0.1)

    def predict_best_action(self, last_state):
        # last_state contains only one state, so we have to convert it into batch of size 1
        last_state.shape = (last_state.shape[0], 1)

        # use neural net to predict Q-values for all actions
        qvalues = self.nnet.predict(last_state)
        #print "Predicted action Q-values: ", qvalues

        # return action (index) with maximum Q-value
        return np.argmax(qvalues)

    def train_minibatch(self, minibatch):
        """
        Train function that transforms (state,action,reward,state) into (input, expected_output) for neural net
        and trains the network
        @param minibatch: list of arrays: prestates, actions, rewards, poststates
        """
        prestates, actions, rewards, poststates = minibatch

        # predict Q-values for prestates, so we can keep Q-values for other actions unchanged
        qvalues = self.nnet.predict(prestates)
        #print "Prestate q-values: ", qvalues[0,:]
        #print "Action was: %d, reward was %d" % (actions[0], rewards[0])

        # predict Q-values for poststates
        post_qvalues = self.nnet.predict(poststates)
        #print "Poststate q-values: ", post_qvalues[0,:]

        # take maximum Q-value of all actions
        max_qvalues = np.max(post_qvalues, axis = 1)

        # update the Q-values for the actions we actually performed
        for i, action in enumerate(actions):
            qvalues[i][action] = rewards[i] + self.discount_factor * max_qvalues[i]
        #print "Corrected q-values: ", qvalues[0,:]

        # we have to transpose prediction result, as train expects input in opposite order
        cost = self.nnet.train(prestates, qvalues.transpose().copy())

        #qvalues = self.nnet.predict(prestates)
        #print "After training: ", qvalues[0,:]

        return cost

    def play_games(self, nr_frames, train, epsilon = None):
        """
        Main cycle: starts a game and plays number of frames.
        @param nr_frames: total number of games allowed to play
        @param train: true or false, whether to do training or not
        @param epsilon: fixed epsilon, only used when not training
        """
        assert train or epsilon is not None

        frames_played = 0
        game_scores = []

        # Start a new game
        self.ale.new_game()
        game_score = 0

        # Play games until maximum number is reached
        while frames_played < nr_frames:

            # Epsilon decreases over time only when training
            if train:
                epsilon = self.compute_epsilon(self.total_frames_trained)
                #print "Current annealed epsilon is %f at %d frames" % (epsilon, self.total_frames_trained)

            # Some times random action is chosen
            if random.uniform(0, 1) < epsilon:
                action = random.choice(range(self.number_of_actions))
                #print "Chose random action %d" % action
            # Usually neural net chooses the best action
            else:
                action = self.predict_best_action(self.memory.get_last_state())
                #print "Neural net chose action %d" % int(action)

            # Make the move
            points = self.ale.move(action)
            if points > 0:
                print "    Got %d points" % points
            game_score += points
            frames_played += 1
            #print "Played frame %d" % frames_played

            # Only if training
            if train:
                # Store new information to memory
                self.ale.store_step(action)
                # Increase total frames only when training
                self.total_frames_trained += 1
                # Fetch random minibatch from memory
                minibatch = self.memory.get_minibatch(self.minibatch_size)
                # Train neural net with the minibatch
                self.train_minibatch(minibatch)
                #print "Trained minibatch of size %d" % self.minibatch_size

            # Play until game is over
            if self.ale.game_over:
                print "    Game over, score = %d" % game_score
                # After "game over" increase the number of games played
                game_scores.append(game_score);
                game_score = 0
                # And do stuff after end game
                self.ale.end_game()
                self.ale.new_game()

        # reset the game just in case
        self.ale.end_game()

        return game_scores

    def run(self, epochs, training_frames, testing_frames):
        # Open log files and write headers
        timestamp = time.strftime("%Y-%m-%d-%H-%M")
        log_train = open("../log/training_" + timestamp + ".csv", "w")
        log_train.write("epoch,nr_games,sum_score,average_score,nr_frames,total_frames_trained,epsilon,memory_size\n")
        log_test = open("../log/testing_" + timestamp + ".csv", "w")
        log_test.write("epoch,nr_games,sum_score,average_score,average_qvalue,nr_frames,epsilon,memory_size\n")
        log_train_scores = open("../log/training_scores_" + timestamp + ".txt", "w")
        log_test_scores = open("../log/testing_scores_" + timestamp + ".txt", "w")
        log_weights = open("../log/weights_" + timestamp + ".csv", "w")

        for epoch in range(1, epochs + 1):
            print "Epoch %d:" % epoch
            # play number of frames with training and epsilon annealing
            print "  Training for %d frames" % training_frames
            training_scores = self.play_games(training_frames, train = True)

            # log training scores
            log_train_scores.write(NL.join(map(str, training_scores)) + NL)
            log_train_scores.flush()

            # log aggregated training data
            train_data = (epoch, len(training_scores), sum(training_scores), np.mean(training_scores), training_frames, self.total_frames_trained, self.compute_epsilon(self.total_frames_trained), self.memory.count)
            log_train.write(','.join(map(str, train_data)) + NL)
            log_train.flush()

            weights = self.nnet.get_weight_stats()
            if epoch == 1:
                # write header
                wlayers = []
                for (layer, index) in weights:
                    wlayers.extend([layer, index, ''])
                log_weights.write(','.join(wlayers) + NL)
                wlabels = []
                for (layer, index) in weights:
                    wlabels.extend(['weights', 'weightsInc', 'incRatio'])
                log_weights.write(','.join(wlabels) + NL)
            wdata = []
            for w in weights.itervalues():
                wdata.extend([str(w[0]), str(w[1]), str(w[1] / w[0] if w[0] > 0 else 0)])
            log_weights.write(','.join(wdata) + NL)
            log_weights.flush()

            # save network state
            self.nnet.save_network(epoch)
            print   # save_network()'s output doesn't include newline

            # play number of frames without training and without epsilon annealing
            print "  Testing for %d frames" % testing_frames
            testing_scores = self.play_games(testing_frames, train = False, epsilon = self.test_epsilon)

            # log testing scores
            log_test_scores.write(NL.join(map(str, testing_scores)) + NL)
            log_test_scores.flush()

            # Pick random states to calculate Q-values for
            if self.random_states is None:
                print "  Picking %d random states for Q-values" % self.nr_random_states
                self.random_states = self.memory.get_minibatch(self.nr_random_states)[0]

            # calculate Q-values 
            qvalues = self.nnet.predict(self.random_states)
            assert qvalues.shape[0] == self.nr_random_states
            assert qvalues.shape[1] == self.number_of_actions
            max_qvalues = np.max(qvalues, axis = 1)
            assert max_qvalues.shape[0] == self.nr_random_states
            assert len(max_qvalues.shape) == 1
            avg_qvalue = np.mean(max_qvalues)

            # log aggregated testing data
            test_data = (epoch, len(testing_scores), sum(testing_scores), np.mean(testing_scores), avg_qvalue, testing_frames, self.test_epsilon, self.memory.count)
            log_test.write(','.join(map(str, test_data)) + NL)
            log_test.flush()

        log_train.close()
        log_test.close()
        log_train_scores.close()
        log_test_scores.close()
        log_weights.close()

if __name__ == '__main__':
    # ignore cuda-convnet options at start of command line
    i = 1
    while (i < len(sys.argv) and sys.argv[i].startswith('--')):
        i += 2

    # take some parameters from command line, otherwise use defaults
    epochs = int(sys.argv[i]) if len(sys.argv) > i else 100
    training_frames = int(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 50000
    testing_frames = int(sys.argv[i + 2]) if len(sys.argv) > i + 2 else 10000

    # run the main loop
    m = Main()
    m.run(epochs, training_frames, testing_frames)
