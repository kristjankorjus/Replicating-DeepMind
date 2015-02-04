"""

This is the main class where all thing are put together

"""

from ai.NeuralNet import NeuralNet
from ai.cnn_q_learner import CNNQLearner
from memory.memoryd import MemoryD
from memory.ale_data_set import DataSet
from ale.ale import ALE
import random
import numpy as np
import time
import sys
from os import linesep as NL
import unittest
import PIL


class Main:
    # How many transitions to keep in memory?
    memory_size = 1000000

    # Size of the mini-batch, 32 was given in the paper
    minibatch_size = 32

    # Number of possible actions in a given game, 6 for "Breakout"
    number_of_actions = 18

    preprocess_type = "cropped_80"

    #image width/height
    if preprocess_type == "article":
        image_size = 84
    else:
        image_size = 80

    # Size of one frame
    frame_size = image_size*image_size

    # Size of one state is four 80x80 screens
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
    nr_random_states = 1000

    # Random states that we use to calculate Q-values
    random_states = None

    # Memory itself
    memory = None

    # Neural net
    nnet = None

    # Communication with ALE
    ale = None

    # The last 4 frames the system has seen
    current_state = None

    def __init__(self):
        #self.memory = MemoryD(self.memory_size)
        self.memory = DataSet(self.image_size, self.image_size, self.memory_size, 4)
        self.ale = ALE(display_screen="true", skip_frames=4, game_ROM='../libraries/ale/roms/breakout.bin', preprocess_type=self.preprocess_type)
        #self.nnet = NeuralNet(self.state_size, self.number_of_actions, "ai/deepmind-layers.cfg", "ai/deepmind-params.cfg", "layer4", discount_factor= self.discount_factor)
        self.nnet = CNNQLearner(self.number_of_actions, 4, self.image_size, self.image_size, discount=self.discount_factor, learning_rate=.0001, batch_size=32, approximator='none')

    def compute_epsilon(self, frames_played):
        """
        From the paper: "The behavior policy during training was epsilon-greedy
        with annealed linearly from 1 to 0.1 over the first million frames, and fixed at 0.1 thereafter."
        @param frames_played: How far are we with our learning?
        """
        return max(0.9 - frames_played / self.epsilon_frames, 0.1)

    def predict_best_action(self, last_state):

        # Uncomment this to see the 4 images that go into q_vals function
        #a = np.hstack(last_state)
        #img = PIL.Image.fromarray(a)
        #img.convert('RGB').save('input_to_nnet.Qvals.png')

        # use neural net to predict Q-values for all actions
        qvalues = self.nnet.q_vals(last_state)
        print "Predicted action Q-values: ", qvalues ,"\n best action is", np.argmax(qvalues)

        # return action (index) with maximum Q-value
        return np.argmax(qvalues)

    def train_minibatch(self, prestates, actions, rewards, poststates):
        """
        Train function that transforms (state,action,reward,state) into (input, expected_output) for neural net
        and trains the network
        @param minibatch: list of arrays: prestates, actions, rewards, poststates
        """

        cost = self.nnet.train(prestates, actions, rewards, poststates)
        #print "trained network, the network thinks cost is: ", type(cost), np.shape(cost), cost

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
        last_frame = self.ale.new_game()

        # We need to initialize/update the current state
        self.current_state = [last_frame.copy(),last_frame.copy(),last_frame.copy(),last_frame.copy()]

        game_score = 0

        # Play games until maximum number is reached
        while frames_played < nr_frames:

            # Epsilon decreases over time only when training
            if train:
                epsilon = self.compute_epsilon(self.total_frames_trained)

            # Some times random action is chosen
            if random.uniform(0, 1) < epsilon or frames_played < 4:
                action = random.choice(range(self.number_of_actions))

            # Usually neural net chooses the best action
            else:
                action = self.predict_best_action(self.current_state)

            # Make the move. Returns points received and the new state
            points, next_frame = self.ale.move(action)

            # Changing points to rewards
            if points > 0:
                print "    Got %d points" % points
                reward = 1
            else:
                reward = 0

            # Book keeping
            game_score += points
            frames_played += 1

            # We need to update the current state
            self.current_state = self.current_state[1:]+[next_frame]

            # Only if training
            if train:

                # Store new information to memory
                self.memory.add_sample(last_frame, action, reward, self.ale.game_over)
                last_frame = next_frame

                if self.memory.count >= self.minibatch_size:
                    # Fetch random minibatch from memory
                    prestates, actions, rewards, poststates, terminals = self.memory.get_minibatch(self.minibatch_size)


                    # Uncomment this to save the minibatch as an image every time we train
                    #b = []
                    #for a in prestates:
                    #    b.append(np.hstack(a))
                    #c = np.vstack(b)
                    #img = PIL.Image.fromarray(c)
                    #img.convert("RGB").save("minibatch.png")

                    # Train neural net with the minibatch
                    self.train_minibatch(prestates, actions, rewards, poststates)

                    # Increase total frames only when training
                    self.total_frames_trained += 1

            # Play until game is over
            if self.ale.game_over:
                print "    Game over, score = %d" % game_score
                # After "game over" increase the number of games played
                game_scores.append(game_score)
                game_score = 0

                # And do stuff after end game
                self.ale.end_game()

                last_frame = self.ale.new_game()

                # We need to update the current state
                self.current_state = self.current_state[1:]+[last_frame]

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

            if training_frames > 0:
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


            if testing_frames > 0:
                # play number of frames without training and without epsilon annealing
                print "  Testing for %d frames" % testing_frames
                testing_scores = self.play_games(testing_frames, train = False, epsilon = self.test_epsilon)

                # log testing scores
                log_test_scores.write(NL.join(map(str, testing_scores)) + NL)
                log_test_scores.flush()

                # Pick random states to calculate Q-values for
                if self.random_states is None and self.memory.count > self.nr_random_states:
                    print "  Picking %d random states for Q-values" % self.nr_random_states
                    self.random_states = self.memory.get_minibatch(self.nr_random_states)[0]

                # Do not calculate Q-values when memory is empty
                if self.random_states is not None:
                    # calculate Q-values
                    qvalues = []
                    for state in self.random_states:
                        qvalues.append(self.nnet.q_vals(state))
                    #assert qvalues.shape[0] == self.nr_random_states
                    #assert qvalues.shape[1] == self.number_of_actions
                    max_qvalues = np.max(qvalues, axis = 1)
                    #assert max_qvalues.shape[0] == self.nr_random_states
                    #assert len(max_qvalues.shape) == 1
                    avg_qvalue = np.mean(max_qvalues)
                else:
                    avg_qvalue = 0

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

    # Running unit tests
    suite = unittest.TestLoader().discover('.')
    unittest.TextTestRunner(verbosity=1).run(suite)

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