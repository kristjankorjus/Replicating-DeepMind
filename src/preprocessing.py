# input: PIL Image object img, dimensions 210x160
# output: PIL image object, dimensions 110x84
def preprocessImage(img):

    # crop to 160x160
    roi = (0, 33, 160, 193) # region of interest (roi) is lines 33 to 193
    img = img.crop(roi)

    # downscale to 84x84
    newSize = 84, 84
    img.thumbnail(newSize) # resizing step
    img.show()

    return img
