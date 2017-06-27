import unittest
from PIL import Image

from model_utils import reduce_opacity


class ModelUtilsTest(unittest.TestCase):
    def test_reduce_opacity(self):
        img = Image.open("./red_pixels_test.png")
        img = reduce_opacity(img, 0.5)
        img.save("trancparency.png")
