import random

import unittest

from img_loader import DatasetLoader


class DatasetLoaderTest(unittest.TestCase):
    def test(self):
        random.seed(123)
        dataset_loader = DatasetLoader("./dataset_test", 10000)
        self.assertEqual(dataset_loader.number_of_imgs, 6)
        self.assertEqual(dataset_loader.nb_classes, 2)
        files = ['0a769a71-052a-4f19-a4d8-b0f0cb75541c___FREC_Scab 3165.JPG',
                                                       '0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417_final_masked.jpg',
                                                       '0b8dabb7-5f1b-4fdc-b3fa-30b289707b90___JR_FrgE.S 3047.JPG',
                                                       '0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG',
                                                       '0a769a71-052a-4f19-a4d8-b0f0cb75541c___FREC_Scab 3165_final_masked.jpg',
                                                       '0b37761a-de32-47ee-a3a4-e138b97ef542___JR_FrgE.S 2908.JPG']
        for i, img_path in enumerate(dataset_loader.imgDataArray):
            self.assertEqual(img_path.name, files[i])
