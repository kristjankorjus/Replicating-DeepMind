"""

This is the main class where all thing are put together

"""

from ai.NeuralNet import NeuralNet
from memory.memoryd import MemoryD
from ale.ale import ALE
import random
import numpy as np
import time
import cPickle

class Main:
    # How many transitions to keep in memory?
    memory_size = 1000000

    # Size of the mini-batch, 32 was given in the paper
    minibatch_size = 32

    # Number of possible actions in a given game, 4 for "Breakout"
    number_of_actions = 4

    # Size of one frame
    frame_size = 84*84

    # How many frames form a history
    history_length = 4

    # Size of one state is four 84x84 screens
    state_size = history_length * frame_size

    # Discount factor for future rewards
    discount_factor = 0.9

    # How many frames to play to choose random frame
    init_frames = 1000
    
    # How many epochs to run
    epochs = 200

    # Number of frames to play during one training epoch
    training_frames = 50000

    # Number of frames to play during one testing epoch
    testing_frames = 10000

    # Exploration rate annealing speed
    epsilon_frames = 1000000.0

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
        assert last_state.shape[0] == self.state_size
        assert len(last_state.shape) == 1

        # last_state contains only one state, so we have to convert it into batch of size 1
        last_state.shape = (last_state.shape[0], 1)
        qvalues = self.nnet.predict(last_state)
        assert qvalues.shape[0] == 1
        assert qvalues.shape[1] == self.number_of_actions
        #print "Predicted action Q-values: ", qvalues

        # return action (index) with maximum Q-value
        return np.argmax(qvalues)

    def train_minibatch(self, minibatch):
        """
        Train function that transforms (state,action,reward,state) into (input, expected_output) for neural net
        and trains the network
        @param minibatch: list of arrays: prestates, actions, rewards, poststates
        """
        prestates = minibatch[0]
        actions = minibatch[1]
        rewards = minibatch[2]
        poststates = minibatch[3]

        assert prestates.shape[0] == self.state_size
        assert prestates.shape[1] == self.minibatch_size
        assert poststates.shape[0] == self.state_size
        assert poststates.shape[1] == self.minibatch_size
        assert actions.shape[0] == self.minibatch_size
        assert rewards.shape[0] == self.minibatch_size

        # predict Q-values for poststates
        post_qvalues = self.nnet.predict(poststates)
        assert post_qvalues.shape[0] == self.minibatch_size
        assert post_qvalues.shape[1] == self.number_of_actions

        # take maximum Q-value of all actions
        max_qvalues = np.max(post_qvalues, axis=1)
        assert max_qvalues.shape[0] == self.minibatch_size
        assert len(max_qvalues.shape) == 1

        # predict Q-values for prestates, so we can keep Q-values for other actions unchanged
        qvalues = self.nnet.predict(prestates)
        assert qvalues.shape[0] == self.minibatch_size
        assert qvalues.shape[1] == self.number_of_actions

        # update the Q-values for the actions we actually performed
        for i, action in enumerate(actions):
            qvalues[i][action] = rewards[i] + self.discount_factor * max_qvalues[i]

        # we have to transpose prediction result, as train expects input in opposite order
        cost = self.nnet.train(prestates, qvalues.transpose().copy())
        return cost

    def play_games(self, nr_frames, train, epsilon):
        """
        Main cycle: starts a game and plays number of frames.
        @param nr_frames: total number of games allowed to play
        @param train: true or false, whether to do training or not
        @param epsilon: fixed epsilon, only used when not training
        """

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
            reward = self.ale.move(action)
            if reward:
                print "    Got reward of %d!!!" % reward
                reward = 1
            game_score += reward
            frames_played += 1
            #print "Played frame %d" % frames_played

            # Store new information to memory
            self.ale.store_step(action)

            # Only if training
            if train:
                # Increase total frames only when training
                self.total_frames_trained += 1
                # Train neural net with random minibatch
                minibatch = self.memory.get_minibatch(self.minibatch_size)
                self.train_minibatch(minibatch)
                #print "Trained minibatch of size %d" % self.minibatch_size

            # Play until game is over
            if self.ale.game_over:
                print "   Game over!!! Score = %d" % game_score
                # After "game over" increase the number of games played
                game_scores.append(game_score);
                game_score = 0
                # And do stuff after end game
                self.ale.end_game()
                self.ale.new_game()

        # reset the game just in case
        self.ale.end_game()

        return game_scores

    def run(self):
        # Play number of random games and pick random states to calculate Q-values for
        print "Playing %d games with random policy" % self.init_frames
        self.play_games(self.init_frames, False, 1)
        self.random_states = self.memory.get_minibatch(self.nr_random_states)[0]

        # Open log file and write header
        log_file = open("../log/scores" + time.strftime("%Y-%m-%d-%H-%M") + ".csv", "w")
        log_file.write("epoch,nr_games,sum_score,average_score,nr_frames_tested,average_qvalue,total_frames_trained,epsilon,memory_size\n")

        for epoch in range(1, self.epochs + 1):
            print "Epoch %d:" % epoch
            # play number of frames with training and epsilon annealing
            print "  Training for %d frames" % self.training_frames
            self.play_games(self.training_frames, True, None)
            # play number of frames without training and without epsilon annealing
            print "  Testing for %d frames" % self.testing_frames
            game_scores = self.play_games(self.testing_frames, False, 0.05)

            # calculate Q-values 
            qvalues = self.nnet.predict(self.random_states)
            assert qvalues.shape[0] == self.nr_random_states
            assert qvalues.shape[1] == self.number_of_actions
            max_qvalues = np.max(qvalues, axis=1)
            assert max_qvalues.shape[0] == self.nr_random_states
            assert len(max_qvalues.shape) == 1
            avg_qvalue = np.mean(max_qvalues)

            # calculate average scores
            sum_score = sum(game_scores)
            nr_games = len(game_scores)
            avg_score = np.mean(game_scores)
            epsilon = self.compute_epsilon(self.total_frames_trained)
            
            # log average scores in file
            log_file.write("%d,%d,%f,%f,%d,%f,%d,%f,%d\n" % (epoch, nr_games, sum_score, avg_score, self.testing_frames, avg_qvalue, self.total_frames_trained, epsilon, self.memory.count))
            log_file.flush()

        log_file.close()

if __name__ == '__main__':
    m = Main()
    m.run()
