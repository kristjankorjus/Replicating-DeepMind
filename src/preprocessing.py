# input: PIL Image object img, dimensions 210x160
# output: PIL image object, dimensions 110x84
def preprocessImage(img):

    # crop to 160x160
    roi = (0, 33, 160, 193) # region of interest (roi) is lines 33 to 193
    img = img.crop(roi)

    # downscale to 84x84
    newSize = 84, 84
    img.thumbnail(newSize) # resizing step
    #img.show()             # show image

    return img


# input: list in the form [img1, a1, img2, a2, ...] where img's are PIL Image objects (210x160) and a's are actions (doesn't really matter what kind of objects
# output: none. THIS FUNCTION MODIFIES THE ORIGINAL LIST OBJECT!
def preprocessSequenceWithActions(sequence):
    from PIL import Image
    for i in xrange(0,len(sequence),2):             # look at 0-th, 2nd, 4th etc element of the list
        ob = sequence[i]
        sequence[i] = preprocessImage(ob)           # preprocess and reassign
    return


# input: list in the form [img1, img2, img3, ...] where img's are PIL Image objects (210x160)
# output: none. THE ORIGINAL LIST IS MODIFIED!
def preprocessSequenceNoActions(sequence):
    from PIL import Image
    for i in range(0,len(sequence)):                # look at all elements in the list
        ob = sequence[i]
        sequence[i] = preprocessImage(ob)           # preprocess and reassign
    return