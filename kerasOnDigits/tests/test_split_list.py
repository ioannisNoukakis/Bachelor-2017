from unittest import TestCase
from imgUtils import split_list


class TestSplit_list(TestCase):
    def test_split_list(self):
        tab = [1, 2, 3, 4]
        t1, t2 = split_list(tab, 4, 0)
        self.assertEqual([1], t1, 'Wrong first array - layout 0')
        self.assertEqual([2, 3, 4], t2, 'Wrong second array - layout 0')

        t1, t2 = split_list(tab, 4, 1)
        self.assertEqual([2], t1, 'Wrong first array - layout 1')
        self.assertEqual([1, 3, 4], t2, 'Wrong second array - layout 1')

        t1, t2 = split_list(tab, 4, 2)
        self.assertEqual([3], t1, 'Wrong first array - layout 2')
        self.assertEqual([1, 2, 4], t2, 'Wrong second array - layout 2')

        t1, t2 = split_list(tab, 4, 3)
        self.assertEqual([4], t1, 'Wrong first array - layout 3')
        self.assertEqual([1, 2, 3], t2, 'Wrong second array - layout 3')
