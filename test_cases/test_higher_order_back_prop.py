if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from kdnn import Variable
import unittest


def _numerical_diff_arg_1(f, x, eps=1e-4):
    x_minus = Variable(np.array(x.data - eps))
    x_plus = Variable(np.array(x.data + eps))
    y_minus = f(x_minus)
    y_plus = f(x_plus)
    return (y_plus - y_minus)/(2*eps)


def _numerical_double_diff_arg_1(f, x, eps=1e-4):
    x_minus = Variable(np.array(x.data - eps))
    x_plus = Variable(np.array(x.data + eps))
    xd_minus = _numerical_diff_arg_1(f, x_minus)
    xd_plus = _numerical_diff_arg_1(f, x_plus)
    return (xd_plus - xd_minus)/(2*eps)


def _numerical_triple_diff_arg_1(f, x, eps=1e-4):
    x_minus = Variable(np.array(x.data - eps))
    x_plus = Variable(np.array(x.data + eps))
    xdd_minus = _numerical_double_diff_arg_1(f, x_minus)
    xdd_plus = _numerical_double_diff_arg_1(f, x_plus)
    return (xdd_plus - xdd_minus) / (2 * eps)


class HigherOrderBackProp(unittest.TestCase):
    def compare_numerical_arg_1(self, f, retain_grad, create_graph):
        x = Variable(np.random.rand(1))
        # x = Variable(np.array(2))
        y = f(x)
        y.back_prop(retain_grad, create_graph)
        xd = x.grad
        x.clear_grad()
        xd.back_prop(retain_grad, create_graph)
        expected = _numerical_double_diff_arg_1(f, x)
        flg = np.allclose(x.grad.data, expected.data)
        self.assertTrue(flg)
        if retain_grad:
            flg = np.allclose(y.grad.data, np.ones_like(y.data))
            self.assertTrue(flg)
        else:
            self.assertEqual(y.grad, None)

        xdd = x.grad
        x.clear_grad()
        xdd.back_prop(retain_grad, create_graph)
        expected = _numerical_triple_diff_arg_1(f, x)
        flg = np.allclose(x.grad.data, expected.data)
        self.assertTrue(flg)

    def test_polynomial(self):
        def polynomial(x):
            y = x**4 - 2*x**2
            return y
        self.compare_numerical_arg_1(polynomial, True, True)
        self.compare_numerical_arg_1(polynomial, False, True)
        # self.compare_numerical_arg_1(polynomial, True, False)
        # self.compare_numerical_arg_1(polynomial, False, False)


if __name__ == '__main__':
    unittest.main()
