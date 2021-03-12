import cv2
import numpy as np
import os
import os.path

class HSV:
    def __init__(self, image):
        self.main(image)

    def get_mean_h(self):
        """return mean h

        Returns:
            float -- nilai mean h
        """


        return self.mean_h

    def get_mean_s(self):
        """return mean s

        Returns:
            float -- nilai mean s
        """


        return self.mean_s

    def get_mean_v(self):
        """return mean v

        Returns:
            float -- nilai mean v
        """


        return self.mean_v

    def read_image(self, image):
        """proses membersihkan image agar sesuai dengan spesifikasi

        Returns:
            numpy -- image array bertype numpy
        """
        if  isinstance(image, str):
            path = image
            if os.path.exists((path)):
                image = cv2.imread(image)
                return image
            else:
                raise FileNotFoundError

        return image

    def rgb_to_hsv(self, R, G, B):
        """rgb to hsv hitung satu piksel

        Arguments:
            R {int} -- nilai R
            G {int} -- nilai G
            B {int} -- nilai B
        """

        R = int(R)
        G = int(G)
        B = int(B)
        R_sum_G_sum_B = R + G + B

        norm_R = R / R_sum_G_sum_B
        norm_G = G/ R_sum_G_sum_B
        norm_B = B/ R_sum_G_sum_B

        V = max(norm_R, norm_G, norm_B)

        S = 0
        if V > 0:
            S = 1 - (min(norm_R, norm_G, norm_B) / V)

        H = 0
        if (S == 0):
            H = 0
        elif (V == norm_R):
            H = (60 * (norm_G - norm_B)) / (S * V)
        elif (V == norm_G):
            H = 60 * (2 + ((norm_B - norm_R) / (S * V)))
        elif (V == norm_B):
            H = 60 * (4 + ((norm_R - norm_G) / (S * V)))

        return H, S, V


    def calculate(self, bgr_image):
        """proses menghitung mean h,s,v pada suatu gambar

        Arguments:
            rgb_image {numpy array} -- image yang akan diproses, urutan matriks R,G,B
        """
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rows, cols, _ = rgb_image.shape

        hsv_image = np.zeros([rows, cols, 3])

        for row in range(0,rows):
            for col in range(0, cols):
                R = rgb_image[row, col][0]
                G = rgb_image[row, col][1]
                B = rgb_image[row, col][2]

                h, s, v = self.rgb_to_hsv(R, G, B)

                hsv_image[row, col, 0] = h
                hsv_image[row, col, 1] = s
                hsv_image[row, col, 2] = v

        self.h = hsv_image[:, :, 0]
        self.s = hsv_image[:, :, 1]
        self.v = hsv_image[:, :, 2]

        self.mean_h = np.mean(hsv_image[:,:,0])
        self.mean_s = np.mean(hsv_image[:,:,1])
        self.mean_v = np.mean(hsv_image[:,:,2])


    def main(self, image):
        clean_image = self.read_image(image)
        self.calculate(clean_image)


if __name__ == '__main__':
     path = r'C:\Users\Popeye\Documents\ILHAM RAHMADHANI\gaharu\fe46c7e0-4a1b-4a5a-ad91-68f399185aef.png'
     a = HSV(path)
     print(a.get_mean_h())
     print(a.get_mean_s())
     print(a.get_mean_v())
