import unittest
from PIL import Image

from model_utils import reduce_opacity


class ModelUtilsTest(unittest.TestCase):
    def test_reduce_opacity(self):
        img = Image.open("./4.png")
        img = reduce_opacity(img, 0.3)
        img.save("trancparency.png")
