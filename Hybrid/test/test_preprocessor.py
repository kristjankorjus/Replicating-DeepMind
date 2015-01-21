# Make sure Python searches the src directory when importing modules
import os, sys, pylab
import matplotlib.pyplot as plt
#import matplotlib.colors.Colormap as col
sys.path.insert(0, os.path.abspath('..\src'))



# This import is probably showing an error if you're using an IDE but the import works.
from ale.preprocessor import Preprocessor
from PIL import Image
import numpy as np

# Create new preprocessor object
pre = Preprocessor()

# Read in sample pixels
f = open("testdata/pixels.dat", "r")
pixs = f.readline()
f.close()

# Preprocess image
img = pre.process(pixs)

# Show the resulting array
# the color mapping is not correct?
plt.imshow(img, cmap=pylab.get_cmap("Purples"), interpolation='nearest')
plt.show()