"""

Preprocessor takes images from ALE and turns them into cropped, downscaled arrays of grayscale values.

"""

from PIL import Image
import numpy as np
import cv2
import scipy

class Preprocessor:

    grayscale_array = None
    desired_image_size = 80         # the size of the new image will be desired_image_size x desired_image_size

    def __init__(self):
        """
        Initialise preprocessor
        """
        self.grayscale_array = self.get_grayscale_array()
        self.NTSC = self.ALE_NTSC_palette()

    def process(self, image_string):
        """
        Returns the cropped, downscaled, grayscale array representation of the image.
        @param image_string: a string that ALE outputs, corresponding to a 160x210 color image
        """
        arr = self.grayscale_array

        # Crop irrelevant lines from beginning and end
        #cropped = image_string[160*33*2:160*193*2]
        cropped = image_string

        # Split cropped image string into a list of hex codes
        hexs = [cropped[i*2:i*2+2] for i in range(len(cropped)/2)]
        colors = np.asarray(map(lambda hex_val: self.NTSC[int(hex_val, 16)], hexs))

        r = []
        g = []
        b = []
        rgb = []
        for cc in colors:
            r.append((cc >> 16) & 0xff)
            g.append((cc >> 8) & 0xff)
            b.append(cc & 0xff)
            rgb.append([r[-1], g[-1], b[-1]])

        rgb = np.array(rgb)
        rgb = rgb.reshape(210, 160, 3)

        gray = np.mean(rgb, axis=2)
        sum_rows = gray[0::2,:] + gray[0::2,:]
        sum_columns = sum_rows[:,0::2] + sum_rows[:,1::2]
        print "compressed img", np.shape(sum_columns), np.mean(sum_columns)
        grays = sum_columns/4.0


        # Uncomment this line to save the COLORED image
        #print "rgb shape", np.shape(rgb)
        #scipy.misc.imsave('our_best_outfile.jpg', rgb)
        #print "print mean rgb", np.mean(rgb)

        # Map each element of the list to the corresponding gray value
        #grays = np.asarray(map(lambda hex_val: arr[int(hex_val[1], 16) ,int(hex_val[0], 16)], hexs))
        #grays = grays.reshape((210, 160))

        # force the 80*105 image to 80*80 using cv2 as Nathan does
        resize_width = 80
        resize_height = 80
        new_size = resize_width, resize_height

        resized = cv2.resize(grays, new_size, interpolation=cv2.INTER_LINEAR)
        resized = np.array(resized, dtype='uint8')

        img = Image.fromarray(resized)
        img.convert('RGB').save('preprocessed.png')
        return resized

    def get_grayscale_array(self):
        """
        Returns the (numpy) array that is used for mapping NTSC colors to grayscale values
        """
        
        my_array = np.array(
            [[0.000000000000000000e+00, 4.533333333333333570e+01, 5.066666666666666430e+01, 5.200000000000000000e+01, 4.533333333333333570e+01, 7.066666666666667140e+01, 6.400000000000000000e+01, 5.066666666666666430e+01, 4.533333333333333570e+01, 4.933333333333333570e+01, 4.533333333333333570e+01, 3.466666666666666430e+01, 2.000000000000000000e+01, 2.533333333333333215e+01, 3.066666666666666785e+01, 3.600000000000000000e+01],
            [6.400000000000000000e+01, 7.200000000000000000e+01, 7.333333333333332860e+01, 7.600000000000000000e+01, 7.333333333333332860e+01, 9.600000000000000000e+01, 9.066666666666667140e+01, 7.733333333333332860e+01, 7.200000000000000000e+01, 7.600000000000000000e+01, 7.466666666666667140e+01, 6.400000000000000000e+01, 5.200000000000000000e+01, 5.733333333333333570e+01, 6.133333333333333570e+01, 6.533333333333332860e+01],
            [1.080000000000000000e+02, 1.000000000000000000e+02, 9.466666666666667140e+01, 1.000000000000000000e+02, 9.866666666666667140e+01, 1.186666666666666714e+02, 1.146666666666666714e+02, 1.026666666666666714e+02, 9.866666666666667140e+01, 1.026666666666666714e+02, 1.013333333333333286e+02, 9.333333333333332860e+01, 8.400000000000000000e+01, 8.666666666666667140e+01, 8.933333333333332860e+01, 9.466666666666667140e+01],
            [1.440000000000000000e+02, 1.240000000000000000e+02, 1.173333333333333286e+02, 1.226666666666666714e+02, 1.226666666666666714e+02, 1.400000000000000000e+02, 1.373333333333333428e+02, 1.280000000000000000e+02, 1.213333333333333286e+02, 1.266666666666666714e+02, 1.280000000000000000e+02, 1.213333333333333286e+02, 1.133333333333333286e+02, 1.133333333333333286e+02, 1.160000000000000000e+02, 1.200000000000000000e+02],
            [1.760000000000000000e+02, 1.440000000000000000e+02, 1.346666666666666572e+02, 1.426666666666666572e+02, 1.440000000000000000e+02, 1.600000000000000000e+02, 1.586666666666666572e+02, 1.480000000000000000e+02, 1.426666666666666572e+02, 1.480000000000000000e+02, 1.506666666666666572e+02, 1.440000000000000000e+02, 1.373333333333333428e+02, 1.386666666666666572e+02, 1.413333333333333428e+02, 1.426666666666666572e+02],
            [2.000000000000000000e+02, 1.653333333333333428e+02, 1.520000000000000000e+02, 1.613333333333333428e+02, 1.653333333333333428e+02, 1.773333333333333428e+02, 1.773333333333333428e+02, 1.693333333333333428e+02, 1.626666666666666572e+02, 1.666666666666666572e+02, 1.720000000000000000e+02, 1.680000000000000000e+02, 1.626666666666666572e+02, 1.613333333333333428e+02, 1.640000000000000000e+02, 1.653333333333333428e+02],
            [2.200000000000000000e+02, 1.853333333333333428e+02, 1.680000000000000000e+02, 1.773333333333333428e+02, 1.853333333333333428e+02, 1.946666666666666572e+02, 1.960000000000000000e+02, 1.880000000000000000e+02, 1.813333333333333428e+02, 1.866666666666666572e+02, 1.933333333333333428e+02, 1.880000000000000000e+02, 1.853333333333333428e+02, 1.840000000000000000e+02, 1.840000000000000000e+02, 1.866666666666666572e+02],
            [2.360000000000000000e+02, 2.026666666666666572e+02, 1.853333333333333428e+02, 1.960000000000000000e+02, 2.040000000000000000e+02, 2.120000000000000000e+02, 2.133333333333333428e+02, 2.066666666666666572e+02, 2.000000000000000000e+02, 2.053333333333333428e+02, 2.133333333333333428e+02, 2.093333333333333428e+02, 2.066666666666666572e+02, 2.053333333333333428e+02, 2.053333333333333428e+02, 2.053333333333333428e+02]]
        )
        return my_array

    def ALE_NTSC_palette(self):
        ourNTSCPalette = [0x000000, 0, 0x4a4a4a, 0, 0x6f6f6f, 0, 0x8e8e8e, 0,
                          0xaaaaaa, 0, 0xc0c0c0, 0, 0xd6d6d6, 0, 0xececec, 0,
                          0x484800, 0, 0x69690f, 0, 0x86861d, 0, 0xa2a22a, 0,
                          0xbbbb35, 0, 0xd2d240, 0, 0xe8e84a, 0, 0xfcfc54, 0,
                          0x7c2c00, 0, 0x904811, 0, 0xa26221, 0, 0xb47a30, 0,
                          0xc3903d, 0, 0xd2a44a, 0, 0xdfb755, 0, 0xecc860, 0,
                          0x901c00, 0, 0xa33915, 0, 0xb55328, 0, 0xc66c3a, 0,
                          0xd5824a, 0, 0xe39759, 0, 0xf0aa67, 0, 0xfcbc74, 0,
                          0x940000, 0, 0xa71a1a, 0, 0xb83232, 0, 0xc84848, 0,
                          0xd65c5c, 0, 0xe46f6f, 0, 0xf08080, 0, 0xfc9090, 0,
                          0x840064, 0, 0x97197a, 0, 0xa8308f, 0, 0xb846a2, 0,
                          0xc659b3, 0, 0xd46cc3, 0, 0xe07cd2, 0, 0xec8ce0, 0,
                          0x500084, 0, 0x68199a, 0, 0x7d30ad, 0, 0x9246c0, 0,
                          0xa459d0, 0, 0xb56ce0, 0, 0xc57cee, 0, 0xd48cfc, 0,
                          0x140090, 0, 0x331aa3, 0, 0x4e32b5, 0, 0x6848c6, 0,
                          0x7f5cd5, 0, 0x956fe3, 0, 0xa980f0, 0, 0xbc90fc, 0,
                          0x000094, 0, 0x181aa7, 0, 0x2d32b8, 0, 0x4248c8, 0,
                          0x545cd6, 0, 0x656fe4, 0, 0x7580f0, 0, 0x8490fc, 0,
                          0x001c88, 0, 0x183b9d, 0, 0x2d57b0, 0, 0x4272c2, 0,
                          0x548ad2, 0, 0x65a0e1, 0, 0x75b5ef, 0, 0x84c8fc, 0,
                          0x003064, 0, 0x185080, 0, 0x2d6d98, 0, 0x4288b0, 0,
                          0x54a0c5, 0, 0x65b7d9, 0, 0x75cceb, 0, 0x84e0fc, 0,
                          0x004030, 0, 0x18624e, 0, 0x2d8169, 0, 0x429e82, 0,
                          0x54b899, 0, 0x65d1ae, 0, 0x75e7c2, 0, 0x84fcd4, 0,
                          0x004400, 0, 0x1a661a, 0, 0x328432, 0, 0x48a048, 0,
                          0x5cba5c, 0, 0x6fd26f, 0, 0x80e880, 0, 0x90fc90, 0,
                          0x143c00, 0, 0x355f18, 0, 0x527e2d, 0, 0x6e9c42, 0,
                          0x87b754, 0, 0x9ed065, 0, 0xb4e775, 0, 0xc8fc84, 0,
                          0x303800, 0, 0x505916, 0, 0x6d762b, 0, 0x88923e, 0,
                          0xa0ab4f, 0, 0xb7c25f, 0, 0xccd86e, 0, 0xe0ec7c, 0,
                          0x482c00, 0, 0x694d14, 0, 0x866a26, 0, 0xa28638, 0,
                          0xbb9f47, 0, 0xd2b656, 0, 0xe8cc63, 0, 0xfce070, 0]
        return ourNTSCPalette
