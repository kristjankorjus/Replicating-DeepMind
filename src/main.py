"""

This is the main class where all thing are put together

"""

from ai.NeuralNet import NeuralNet
from memory.memoryd import MemoryD
from ale.ale import ALE
import random


class Main:
    memory_size = 100000   # how many transitions to keep in memory?
    memory = None
    nnet = None
    ale = None
    minibatch_size = None
    number_of_actions = None

    def __init__(self):
        self.memory = MemoryD(self.memory_size)
        self.minibatch_size = 32
        self.nnet = NeuralNet([self.minibatch_size, 4, 84, 84], filter_shapes=[[16, 4, 8, 8], [32, 16, 4, 4]],
                              strides=[4, 2], n_hidden=256, n_out=4)
        self.ale = ALE(self.memory)
        self.number_of_actions = 4

    def compute_epsilon(self, frames_played):
        """
        From the paper: "The behavior policy during training was epsilon-greedy
        with annealed linearly from 1 to 0.1 over the first million frames, and fixed at 0.1 thereafter."
        """
        return max(1 - frames_played / self.memory_size, 0.1)

    def play_games(self, n):
        """
        Main cycle: plays many games and many frames in each game. Also learning is performed.
        """

        games_to_play = n
        games_played = 0
        frames_played = 0

        while games_played < games_to_play:
            # Start a new game
            self.ale.new_game()

            # Play until game is over
            while not self.ale.game_over:

                # Epsilon decreases over time
                epsilon = self.compute_epsilon(frames_played)

                # Some times random action is chosen
                if random.uniform(0, 1) < epsilon:
                    action = random.choice(range(self.number_of_actions))

                # Usually neural net chooses the best action
                else:
                    action = self.nnet.predict_best_action([self.memory.get_last_state]*32)

                # Make the move
                self.ale.move(action)

                # Store new information to memory
                self.ale.store_step(action)

                # Start a training session
                self.nnet.train(self.memory.get_minibatch(self.minibatch_size))

            # After "game over" increase the number of games played
            games_played += 1

            # And do stuff after end game (store information, let ALE know etc)
            self.ale.end_game()


if __name__ == '__main__':
    m = Main()
    m.play_games(10)