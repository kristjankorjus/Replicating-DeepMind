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

from preprocessing import preprocessImage

preprocessImage(img)