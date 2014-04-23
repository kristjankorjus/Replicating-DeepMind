f = open("pixels.dat", "r")
pixs = f.readline()
f.close()

print len(pixs)

from PIL import Image
import numpy as np

img = Image.new('RGB', (160, 210), "black")  # create a new black image
pixels = img.load()  # create the pixel map
colMat = np.loadtxt("Grayscale.dat")

for i in range(len(pixs)/2):
    row = i % 160
    column = i/160
    hex1 = int(pixs[i*2], 16)

    # Division by 2 because: http://en.wikipedia.org/wiki/List_of_video_game_console_palettes
    hex2 = int(pixs[i*2+1], 16)/2
    temp = int(colMat[hex2, hex1])
    pixels[row, column] = (temp, temp, temp)

img.show()



# Example 1: take one PIL.Image file, preprocess and get its pixel array
from preprocessing import preprocessImage
img2 = preprocessImage(img)
pixels = img2.load()

# Example 2: take a sequence that DOESN'T contain actions and preprocess the images in-place
from preprocessing import preprocessSequenceWithActions
sequence = [img.copy(), 45, img.copy(), 'thisdoesntmatter', img.copy(), 'this neither'] #,deepcopy(img),'thisdoesntmatter',deepcopy(img),deepcopy(img)]
sequence = preprocessSequenceWithActions(sequence)

# Example 3: take a sequence that DOES contain actions and preprocess the images in-place
from preprocessing import preprocessSequenceNoActions
sequence = [img.copy(), img.copy(), img.copy()]
sequence = preprocessSequenceNoActions(sequence)
