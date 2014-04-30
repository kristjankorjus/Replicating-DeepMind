"""

This is the main class where all thing are put together

"""

from ai.neuralnet import NeuralNet

class Main:

    memory = None
    nnet = None
    ale = None

    def __init__(self):
        self.memory = None
        self.nnet = None
        self.ale = None


if __name__ == '__main__':
    m = Main()
    m.playGames(10)