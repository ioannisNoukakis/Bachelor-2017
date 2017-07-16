import colorsys
import unittest

from PIL import Image

from img_processing import pixels_attention_score, merge_images_mask


class PixelsCounterTest(unittest.TestCase):
    def test(self):
        cam = Image.open("./cam.png").convert('RGBA')
        mask = Image.open("./mask.jpg").convert('RGBA')

        cam_a_p, cam_a_e = merge_images_mask(cam, mask)
        score = pixels_attention_score(cam_a_p)
        print(score)


        # bound_upper=(255, 0, 0), bound_lower=(183, 253, 52)
