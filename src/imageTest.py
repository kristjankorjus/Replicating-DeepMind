f = open("pixels.dat", "r")
pixs = f.readline()
f.close()

print len(pixs)

from PIL import Image

img = Image.new('RGB', (160, 210), "black")  # create a new black image
pixels = img.load()  # create the pixel map

for i in range(len(pixs)/2):
    row = i % 160
    column = i/160
    hex = pixs[i*2:(i+1)*2]
    pixels[row, column] = (int(hex, 16), int(hex, 16), int(hex, 16))

img.show()



# Example 1: take one PIL.Image file, preprocess and get its pixel array
from preprocessing import preprocessImage
img2 = preprocessImage(img)
pixels = img2.load()

# Example 2: take a sequence that DOESN'T contain actions and preprocess the images in-place
from preprocessing import preprocessSequenceWithActions
sequence = [img.copy(),45,img.copy(),'thisdoesntmatter',img.copy(),'this neither'] #,deepcopy(img),'thisdoesntmatter',deepcopy(img),deepcopy(img)]
sequence = preprocessSequenceWithActions(sequence)

# Example 3: take a sequence that DOES contain actions and preprocess the images in-place
from preprocessing import preprocessSequenceNoActions
sequence = [img.copy(),img.copy(),img.copy()]
sequence = preprocessSequenceNoActions(sequence)
