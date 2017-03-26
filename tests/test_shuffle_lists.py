from unittest import TestCase
from imgUtils import shuffle_lists
import random


class TestShuffle_lists(TestCase):
    def setUp(self):
        random.seed(123)

    def test_shuffle_lists(self):
        tab1 = [1, 2, 3, 4]
        tab2 = ['a', 'b' , 'c', 'd']
        tabf1, tabf2 = shuffle_lists(tab1, tab2)
        self.assertNotEqual(tab1, tabf1, "Arrays 1 are the same")
        self.assertNotEqual(tab2, tabf2, "Arrays 2 are the same")
