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

    # Memory itself
    memory = None

    # Neural net
    nnet = None

    # Communication with ALE
    ale = None

    # Size of the mini-batch which will be sent to learning in Theano
    minibatch_size = None

    # Number of possible actions in a given game
    number_of_actions = None

    def __init__(self):
        self.memory = MemoryD(self.memory_size)
        self.minibatch_size = 32  # Given in the paper
        self.number_of_actions = 4  # Game "Breakout" has 4 possible actions

        # Properties of the neural net which come from the paper
        self.nnet = NeuralNet()
        self.ale = ALE(self.memory)

    def compute_epsilon(self, frames_played):
        """
        From the paper: "The behavior policy during training was epsilon-greedy
        with annealed linearly from 1 to 0.1 over the first million frames, and fixed at 0.1 thereafter."
        @param frames_played: How far are we with our learning?
        """
        return max(1.0 - frames_played / (1000000 * 1.0), 0.1)

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

        # Play games until maximum number is reached
        while games_played < games_to_play:

            # Start a new game
            self.ale.new_game()
            print "starting game ", games_played+1, " frames played so far: ", frames_played
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
                    action = self.nnet.predict_best_action(self.memory.get_last_state())

                # Make the move
                reward = self.ale.move(action)
                game_score += reward

                # Store new information to memory
                self.ale.store_step(action)

                # Start a training session
                batch = self.memory.get_minibatch(self.minibatch_size)
                self.nnet.train(batch)
                frames_played += 1

            # After "game over" increase the number of games played
            games_played += 1
            
            # Store game state every 100 games
            if games_played % 100 == 0:

                # Store state of the network as cpickle as Convnet does
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

