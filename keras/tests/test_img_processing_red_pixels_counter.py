import unittest

from PIL import Image


def pixels_counter(image: Image, bound_upper, bound_lower):
    (r1, g1, b1) = bound_upper
    (r2, g2, b2) = bound_lower

    score = 0
    nPixels = 0
    newdata = []
    image = image.convert("RGBA")
    for item in image.getdata():
        if r1 >= item[0] >= r2 and g1 <= item[1] <= g2 and b1 <= item[2] <= b2:
            score += 1
            newdata.append(item)
        else:
            newdata.append((0, 0, 0, 0))
        nPixels += 1

    image.putdata(newdata)
    image.show()

    return score/nPixels


class PixelsCounterTest(unittest.TestCase):
    def test(self):
        img = Image.open("./red_pixels_test.png")
        res = pixels_counter(img, bound_upper=(255, 0, 0), bound_lower=(183, 253, 52))
        self.assertTrue(res > 0.5)
