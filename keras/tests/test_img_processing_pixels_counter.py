import colorsys
import unittest

from PIL import Image

from img_processing import pixels_counter


class PixelsCounterTest(unittest.TestCase):
    def test(self):
        img = Image.open("./4.png").convert('RGBA')
        res = pixels_counter(img)
        self.assertTrue(res > 0.8)
        img = Image.open("./5.png").convert('RGBA')
        res = pixels_counter(img)
        self.assertTrue(res < 0.1)
        img = Image.open("./6.png").convert('RGBA')
        res = pixels_counter(img)
        self.assertTrue(0.45 < res < 0.55)

        # bound_upper=(255, 0, 0), bound_lower=(183, 253, 52)
