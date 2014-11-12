"""

Set of functions to explore state of the network given cPickle dump file

"""

import cPickle
import numpy as np


class Analyzer:

    def __init__(self, filename):

        # read the data in
        with open(filename, 'rb') as f:
            self.data = cPickle.load(f)

    def sandbox(self):
        """ The main place to try out things """
        print "Maximal weight", np.max(self.data['layer1']['weights'])
        print "Minimal weight", np.min(self.data['layer1']['weights'])


if __name__ == "__main__":
    analyzer = Analyzer("weights_at_5700_games.pkl")
    analyzer.sandbox()
