import math
import threading
import copy
from PIL import Image

class ImageFilter:
    def __init__(self, image):
        self.image = image
        self.newImage = Image.new("RGB", image.size)
        self.pix = self.newImage.load()
        self.refPix = self.image.load()

    def applyFilterOnImage(self, filtre, nb_threads):
        previous_slice = 1
        slice = math.floor(self.image.size[0] / nb_threads)

        threads = []
        for i in range(nb_threads):
            t = threading.Thread(name=str(i), target=self.workerThread, args=(filtre, previous_slice, slice))
            threads.append(t)
            previous_slice = slice
            slice += math.floor(self.image.size[0] / nb_threads)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        print("FINISHED! saving...")
        self.newImage.save('out.jpg')

    def workerThread(self, filtre, begin, end):
        print("starting", begin, "-", end)
        for i in range(begin, end):
            for j in range(1, self.image.size[1]-1):
                self.applyFilter(filtre, i, j)

        print("THREAD COMPLETED")

    def applyFilter(self, filtre, x, y):
        if x == 0 or y == 0 or x == self.image.size[0] - 1 or y == self.image.size[1] - 1:
            return
        else:
            for i in range(len(filtre)):
                for j in range(len(filtre[i])):
                    a = x + i - math.floor(len(filtre) / 2)
                    b = y + j - math.floor(len(filtre) / 2)
                    try:
                        self.pix[x, y] = self.getRGBfromI(self.getIfromRGB(self.refPix[a, b]) * filtre[i, j])
                    except IndexError:
                        print("OUT OF RANGE! : ", str(i), " ", str(j))

    # TODO: CHECK IF THERE IS NO OVERFLOW
    def getRGBfromI(self, RGBint):
        blue = RGBint & 255
        green = (RGBint >> 8) & 255
        red = (RGBint >> 16) & 255
        return red, green, blue

    def getIfromRGB(self, rgb):
        red = rgb[0]
        green = rgb[1]
        blue = rgb[2]
        RGBint = (red << 16) + (green << 8) + blue
        return RGBint
