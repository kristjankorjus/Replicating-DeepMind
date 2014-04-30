"""

This is the main class where all thing are put together

"""

from ai.neuralnet import NeuralNet
from memory.memoryd import MemoryD
import random

class Main:

    memory = None
    nnet = None
    ale = None

    def __init__(self):
        self.memory = MemoryD(1000000)
        self.nnet = NeuralNet()
        self.ale = None

    def compute_epsilon(self, frames_played):
        """
        From the paper: "The behavior policy during training was epsilon-greedy
        with annealed linearly from 1 to 0.1 over the first million frames, and fixed at 0.1 thereafter."
        """
        return max(1 - frames_played / 1000000, 0.1)

    def play_games(self, n):

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
                    action = random.choice(ale.actions)

                # Usually neural net chooses the best action
                else:
                    action = self.nnet.get_action(memory.get_last_state)

                # Make the move
                self.ale.move(action)

                # Store new information to memory
                self.ale.store_transition()

                # Start a training session
                self.nnet.train()

            # After "game over" increase the number of games played
            games_played++

            # And do stuff after end game (store information, let ALE know etc)
            self.ale.end_game()

if __name__ == '__main__':
    m = Main()
    m.play_games(10)