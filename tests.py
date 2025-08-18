import numpy as np
from process import trim
import unittest


class Tests(unittest.TestCase):
    def test_trim(self):
        x = np.array([300, 200, 500, 400, 100])
        y = np.array([3, 2, 5, 4, 1])
        ref_x = np.array([200, 400])
        result_x, result_y = trim(x, y, ref_x)
        self.assertEqual(list(result_x), [200, 300, 400])
        self.assertEqual(list(result_y), [2, 3, 4])



