
import numpy as np
from PIL import Image




def make_array():
    #result = np.zeros((84,84))

    row1 = np.tile(np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]),(8,6))
    row2 = np.tile(np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]),(8,6))

    tworows = np.vstack((row1, row2))
    #print(str(np.size(tworows,0)) + 'x' + str(np.size(tworows,1)))

    result = np.tile(tworows,(6,1))
    result = result[0:84,0:84]

    return result








def show(result):
    # Display array as image
    img = Image.new('RGB', (84,84), "black")  # create a new black image
    pixels = img.load()  # create the pixel map
    print(str(np.size(result,0)) + 'x' + str(np.size(result,1)))
    for i in range(84):
        for j in range(84):
            val = int(result[i][j])
            pixels[i, j] = (val, val, val)

    img.show()

arr = make_array()
show(arr)
