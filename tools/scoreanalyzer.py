"""

Plot and compare game score trails

"""

import numpy as np
import matplotlib.pylab as plt
from itertools import cycle


class ScoreAnalyzer:

    scores = {}

    def __init__(self, files):

        for filename in files:
            with open(filename, 'rb') as f:
                self.scores[filename] = [int(s.strip()) for s in f.readlines()]

    @staticmethod
    def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    def plot(self):

        # set smoothing parameter
        smoothing = 300

        # enable to loop over different line type
        lines = [":", "-", "--", "-."]
        linecycler = cycle(lines)

        # draw things
        for filename in self.scores.keys():
            plt.plot(self.smooth(self.scores[filename], smoothing)[:-smoothing/2], label=filename,
                     linestyle=next(linecycler))
        plt.legend(loc=4)
        plt.show()

    def sandbox(self):
        """ The main place to try out things """
        print self.scores.keys()
        print self.scores[1]


if __name__ == "__main__":
    analyzer = ScoreAnalyzer(['data/origusegrads.txt',
                              'data/forceusegrads.txt',
                              'data/endstatefix.txt'])
    analyzer.plot()
