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
        scores = self.nnet.predict(last_state)
        assert scores.shape[1] == self.number_of_actions

        self.output_file.write(str(scores).strip().replace(' ', ',')[2:-2] + '\n')
        self.output_file.flush()
        
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
        scores_file = open("../log/scores" + time.strftime("%Y-%m-%d-%H-%M") + ".txt", "w")
        self.output_file = open("../log/Q_history"+time.strftime("%Y-%m-%d-%H-%M")+".csv","w")

        # Play games until maximum number is reached
        while games_played < games_to_play:

            # Start a new game
            self.ale.new_game()
            print "Starting a new game", games_played+1, "frames played so far:", frames_played
            game_score = 0
            self.nnet.epoch = games_played

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
            
            # Store game state every 100 games
            if games_played % 100 == 0:

                # Store state of the network as cpickle as Convnet does
                self.nnet.batchnum = 1
                self.nnet.epoch = 1
                self.nnet.sync_with_host()
                self.nnet.save_state()
            
                # Store the weights and biases of all layers
                layers_list=["layer1","layer2","layer3","layer4"]
                layer_dict = {}
                for layer_name in layers_list:
                    w = m.nnet.layers[layer_name]["weights"][0].copy()
                    b = m.nnet.layers[layer_name]["biases"][0].copy()
                    layer_dict[layer_name] = {'weights': w, 'biases': b}
                filename = "../log/weights_at_" + str(games_played) + "_games.pkl"
                weights_file = open(filename, "wb")
                cPickle.dump(layer_dict, weights_file)
                weights_file.close()

            # write the game score to a file 
            scores_file.write(str(game_score)+"\n")
            scores_file.flush()

            # And do stuff after end game (store information, let ALE know etc)
            self.ale.end_game()

        print game_scores
        scores_file.close()

if __name__ == '__main__':
    m = Main()
    nr_games = 100000
    m.play_games(nr_games)
    #m.nnet.sync_with_host()
    #w1= m.nnet.layers["layer4"]["weights"][0].copy()

