if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import unittest
from kdnn import Variable


class VariableTest(unittest.TestCase):
    def test_operator(self):
        # add
        # Variable + ndarray
        np_data1 = np.random.rand(2)
        np_data2 = np.random.rand(2)
        x = Variable(np.array(np_data1))
        y = np.array(np_data2)
        a = x + y
        flg = np.allclose(a.data, np_data1 + np_data2)
        self.assertTrue(flg)

        # ndarray + Variable
        a = y + x
        flg = np.allclose(a.data, np_data1 + np_data2)
        self.assertTrue(flg)

        # Variable + float
        y = 3.0
        a = x + y
        flg = np.allclose(a.data, np_data1 + y)
        self.assertTrue(flg)

        # float + Variable
        y = 3.0
        a = y + x
        flg = np.allclose(a.data, np_data1 + y)
        self.assertTrue(flg)

        # neg
        x = -x
        np_data1 = -np_data1
        flg = np.allclose(x.data, np_data1)
        self.assertTrue(flg)

        # mul
        y = 3
        a = x*y
        flg = np.allclose(a.data, np_data1*y)
        self.assertTrue(flg)

        # sub
        y = 3.0
        a = x - y
        flg = np.allclose(a.data, np_data1 - y)
        self.assertTrue(flg)
        y = 3.0
        a = y - x
        flg = np.allclose(a.data, y - np_data1)
        self.assertTrue(flg)

        # div
        y = 3.0
        a = x / y
        flg = np.allclose(a.data, np_data1 / y)
        self.assertTrue(flg)
        y = 3.0
        a = y / x
        flg = np.allclose(a.data, y / np_data1)
        self.assertTrue(flg)

        # pow
        y = 3.0
        a = x**y
        flg = np.allclose(a.data, np_data1**y)
        self.assertTrue(flg)

    def test_property_return(self):
        x = Variable(np.array([[1.0, 2, 3], [4, 5, 6]]))
        sp = x.shape
        self.assertEqual(sp[0], 2)
        self.assertEqual(sp[1], 3)

        dim = x.ndim
        self.assertEqual(dim, 2)

        sz = x.size
        self.assertEqual(sz, 6)

        dtp = x.dtype
        self.assertEqual(dtp, np.float64)

        val_t = x.T
        flg = np.allclose(val_t.data, x.data.T)
        self.assertTrue(flg)

        ln = len(x)
        self.assertEqual(ln, 2)


if __name__ == '__main__':
    unittest.main()
