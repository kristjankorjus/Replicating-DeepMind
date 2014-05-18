
import numpy as np
from math import ceil



def make_chessboard_any_size(image_size, n):
    """ Create a image_size x image_size chessboard with square size n pixels.
    """
    # n is length of small square

    n_ones = n * [1.0]
    n_zeros = n * [0.0]

    repx = int(ceil(image_size/(2*n))) + 1
    #print(int(ceil(image_size/(2*n))))

    row1 = np.tile(np.array(n_ones + n_zeros),(n,repx))
    row2 = np.tile(np.array(n_zeros + n_ones),(n,repx))

    tworows = np.vstack((row1, row2))

    result = np.tile(tworows,(repx,1))
    result = result[0:image_size,0:image_size]


    #print('Size: '+ str(np.size(result,0)) + 'x' + str(np.size(result,1)))
    return result


def make_chessboard(n):
    """ Create a 84x84 chessboard with small square size n pixels.
    """
    return make_chessboard_any_size(84, n)






def show(result):
    import matplotlib.pyplot as plt
    plt.imshow(result, interpolation='nearest')
    plt.show()

arr = make_chessboard(8)
show(arr)
