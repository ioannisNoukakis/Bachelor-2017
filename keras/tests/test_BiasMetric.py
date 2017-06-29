import unittest
import tensorflow as tf

from bias_metric import BiasMetric, save_to_csv


class PixelsCounterTest(unittest.TestCase):
    def test(self):
        bm = BiasMetric(tf.get_default_graph())
        bm.l1 = [0.3, 0.6, 0.8]
        bm.l2 = [0.3, 0.2, 0.1]
        bm.e1 = [0.3, 0.3, 0.3]
        bm.e2 = [0.2, 0.2, 0.2]
        bm.metric1 = [0, 0.3, 0.5]
        bm.metric2 = [0.1, 0, -0.1]

        save_to_csv()

        with open('results.csv', 'r') as f:
            content = f.read()
            self.assertEqual(content, "l1,l2,e1,e2,metric1,metric2\n"
                                      "0.3,0.3,0.3,0.2,0,0.1\n"
                                      "0.6,0.2,0.3,0.2,0.3,0\n"
                                      "0.8,0.1,0.3,0.2,0.5,-0.1\n")