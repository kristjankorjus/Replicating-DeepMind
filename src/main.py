"""

This is the main class where all thing are put together

"""

from ai.NeuralNet import NeuralNet
from memory.memoryd import MemoryD
from ale.ale import ALE
import random
import numpy as np
import time

# Definitions needed for The Three Laws
injury_to_a_human_being    = None
conflict_with_orders_given = None
threat_to_my_existence     = None


class Main:
    # How many transitions to keep in memory?
    memory_size = 300000

    # Size of the mini-batch, 32 was given in the paper
    minibatch_size = 32

    # Number of possible actions in a given game, 4 for "Breakout"
    number_of_actions = 4

    # Size of one state is four 84x84 screens
    state_size = 4*84*84

    # Discount factor for future rewards
    discount_factor = 0.9

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
        return max(1.0 - frames_played / (1000000 * 1.0), 0.1)

    def predict_best_action(self, last_state):
        assert last_state.shape[0] == self.state_size
        assert len(last_state.shape) == 1

        # last_state contains only one state, so we have to convert it into batch of size 1
        last_state.shape = (last_state.shape[0], 1)
        scores = self.predict(last_state)
        assert scores.shape[0] == self.number_of_actions

        # return action (index) with maximum score
        return np.argmax(scores)

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

        # predict scores for poststates
        post_scores = self.nnet.predict(poststates)
        assert post_scores.shape[0] == self.minibatch_size
        assert post_scores.shape[1] == self.number_of_actions

        # take maximum score of all actions
        max_scores = np.max(post_scores, axis=1)
        assert max_scores.shape[0] == self.minibatch_size
        assert len(max_scores.shape) == 1

        # predict scores for prestates, so we can keep scores for other actions unchanged
        scores = self.nnet.predict(prestates)
        assert scores.shape[0] == self.minibatch_size
        assert scores.shape[1] == self.number_of_actions

        # update the Q-values for the actions we actually performed
        for i, action in enumerate(actions):
            scores[i][action] = rewards[i] + self.discount_factor * max_scores[i]

        # we have to transpose prediction result, as train expects input in opposite order
        cost = self.nnet.train(prestates, scores.transpose().copy())
        return cost

    def play_games(self, n):
        """
        Main cycle: plays many games and many frames in each game. Also learning is performed.
        @param n: total number of games allowed to play
        """

        games_to_play = n
        games_played = 0
        frames_played = 0
        game_scores = []
        f = open("../log/scores" + time.strftime("%Y-%m-%d-%H-%M") + ".txt", "w")

        # Play games until maximum number is reached
        while games_played < games_to_play:

            # Start a new game
            self.ale.new_game()
            print "starting game", games_played+1, "frames played so far:", frames_played
            game_score = 0

            # Play until game is over
            while not self.ale.game_over:

                # Epsilon decreases over time
                epsilon = self.compute_epsilon(frames_played)

                # Before AI takes an action we must make sure it is safe for the human race
                if   injury_to_a_human_being    is not None:
                    raise Exception('The First Law of Robotics is violated!')
                elif conflict_with_orders_given is not None:
                    raise Exception('The Second Law of Robotics is violated!')
                elif threat_to_my_existence     is not None:
                    raise Exception('The Third Law of Robotics is violated!')

                # Some times random action is chosen
                if random.uniform(0, 1) < epsilon:
                    action = random.choice(range(self.number_of_actions))

                # Usually neural net chooses the best action
                else:
                    action = self.predict_best_action(self.memory.get_last_state())

                # Make the move
                reward = self.ale.move(action)
                game_score += reward

                # Store new information to memory
                self.ale.store_step(action)

                # Start a training session
                minibatch = self.memory.get_minibatch(self.minibatch_size)
                self.train_minibatch(minibatch)
                frames_played += 1

            # After "game over" increase the number of games played
            games_played += 1
            
            # write the game score to a file 
            f.write(str(game_score)+"\n")
            f.flush()

            # And do stuff after end game (store information, let ALE know etc)
            self.ale.end_game()

        print game_scores
        f.close()

if __name__ == '__main__':
    m = Main()
    nr_games = 100000
    m.play_games(nr_games)
    #m.nnet.sync_with_host()
    #w1= m.nnet.layers["layer4"]["weights"][0].copy()

