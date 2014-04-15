# input: PIL Image object img, dimensions 210x160
# output: PIL image object, dimensions 110x84
def preprocessImage(img):
    newSize = 110, 84
    img.thumbnail(newSize) # resizing step
    img.show()

    return img
