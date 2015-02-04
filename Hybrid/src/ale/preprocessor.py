"""

Preprocessor takes images from ALE and turns them into cropped, downscaled arrays of grayscale values.

"""

from PIL import Image
import numpy as np
import cv2
import scipy

class Preprocessor:

    grayscale_array = None
    desired_image_size = None         # the size of the new image will be desired_image_size x desired_image_size

    def __init__(self, preprocess_type):
        """
        Initialise preprocessor
        """

        self.NTSC = self.ALE_NTSC_palette()
        if preprocess_type == "article":
            self.desired_image_size = 84
        elif preprocess_type == "cropped_80":
            self.desired_image_size = 80
            self.cropped = True
        elif preprocess_type == "resized_80":
            self.desired_image_size = 80
            self.cropped = False
        else:
            print "unknown preprocess type"

    def process(self, image_string):
        """
        Returns the cropped, downscaled, grayscale array representation of the image.
        @param image_string: a string that ALE outputs, corresponding to a 160x210 color image
        """

        # Split cropped image string into a list of hex codes,
        # then get the corresponding color_values(integers) from NTSC table
        hexs = [image_string[i*2:i*2+2] for i in range(len(image_string)/2)]
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

        # average over R, G and B
        gray = np.mean(rgb, axis=2)

        if self.desired_image_size == 84:
            resized = cv2.resize(gray, (84, 110), interpolation=cv2.INTER_LINEAR)
            # Nathan suggests to crop 8 lines of his 105.. we have 110 lines
            final_shape = resized[110-84-8:110-8, :]

        else:
            # in the case of using 80x80, we start by averaging 4 pixels as in deep_q_l
            # take the average of 4 pixels
            sum_rows = gray[0::2, :] + gray[0::2,:]
            sum_columns = sum_rows[:, 0::2] + sum_rows[:, 1::2]
            grays = sum_columns/4.0


            if not self.cropped:
            # force the 80*105 image to 80*80 using cv2 as Nathan does
                new_size = self.desired_image_size, self.desired_image_size
                resized = cv2.resize(grays, new_size, interpolation=cv2.INTER_LINEAR)
                final_shape = np.array(resized, dtype='uint8')

            else:
                # The case if choose to crop and not force the image into new shape with OpenCV
                # Nathan suggests that in Breakout we cut off 8 lines from the bottom
                lower_cut_off = 8
                lower_bound = 105 - lower_cut_off
                higher_bound = lower_bound - self.desired_image_size
                final_shape = grays[higher_bound: lower_bound, :]

        # Uncomment this section to save the COLORED image
        #print "rgb shape", np.shape(rgb)
        #scipy.misc.imsave('our_best_outfile.jpg', rgb)
        #print "print mean rgb", np.mean(rgb)

        # Uncomment this section to save the proprocessed image
        #print np.shape(final_shape)
        #img = Image.fromarray(final_shape)
        #img.convert('RGB').save('preprocessed.png')

        return final_shape


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
