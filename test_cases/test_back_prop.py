if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from kdnn import Variable
from kdnn.core_simplified import add, neg, sub, mul, div, rsub, rdiv, pow
from kdnn.functions import square, exp
import unittest


def numerical_diff_arg_1(f, x, eps=1e-4):
    x_minus = Variable(x.data - eps)
    x_plus = Variable(x.data + eps)
    y_minus = f(x_minus)
    y_plus = f(x_plus)
    return (y_plus.data - y_minus.data)/(2*eps)


def numerical_diff_arg_2_0(f, x, eps=1e-4):
    x_minus = Variable(x.data - eps)
    x_plus = Variable(x.data + eps)
    y_minus = f(x_minus, x_minus)
    y_plus = f(x_plus, x_plus)
    return (y_plus.data - y_minus.data)/(2*eps)


def numerical_diff_arg_2_1(f, x0, x1, eps=1e-4):
    x0_minus = Variable(x0.data - eps)
    x0_plus = Variable(x0.data + eps)
    y_minus = f(x0_minus, x1)
    y_plus = f(x0_plus, x1)
    return (y_plus.data - y_minus.data)/(2*eps)


def numerical_diff_arg_2_2(f, x0, x1, eps=1e-4):
    x1_minus = Variable(x1.data - eps)
    x1_plus = Variable(x1.data + eps)
    y_minus = f(x0, x1_minus)
    y_plus = f(x0, x1_plus)
    return (y_plus.data - y_minus.data)/(2*eps)


class BackProp(unittest.TestCase):
    def compare_numerical_arg_1(self, f, retain_grad):
        x = Variable(np.random.rand(1))
        y = f(x)
        y.back_prop(retain_grad)
        expected = numerical_diff_arg_1(f, x)
        flg = np.allclose(x.grad, expected)
        self.assertTrue(flg)
        if retain_grad:
            flg = np.allclose(y.grad, np.ones_like(y.data))
            self.assertTrue(flg)
        else:
            self.assertEqual(y.grad, None)

    def compare_numerical_arg_2(self, f):
        x0 = Variable(np.random.rand(1))
        x1 = Variable(np.random.rand(1))
        y = f(x0, x1)
        y.back_prop(True)

        expected = numerical_diff_arg_2_1(f, x0, x1)
        flg = np.allclose(x0.grad, expected)
        self.assertTrue(flg)
        flg = np.allclose(y.grad, np.ones_like(y.data))
        self.assertTrue(flg)

        expected = numerical_diff_arg_2_2(f, x0, x1)
        flg = np.allclose(x1.grad, expected)
        self.assertTrue(flg)
        flg = np.allclose(y.grad, np.ones_like(y.data))
        self.assertTrue(flg)

        x0 = Variable(np.random.rand(1))
        y = f(x0, x0)
        y.back_prop(True)
        expected = numerical_diff_arg_2_0(f, x0)
        flg = np.allclose(x0.grad, expected)
        self.assertTrue(flg)
        flg = np.allclose(y.grad, np.ones_like(y.data))
        self.assertTrue(flg)

    def test_squared(self):
        self.compare_numerical_arg_1(square, True)
        self.compare_numerical_arg_1(square, False)

    def test_exp(self):
        self.compare_numerical_arg_1(exp, True)
        self.compare_numerical_arg_1(exp, False)

    def test_neg(self):
        self.compare_numerical_arg_1(neg, True)
        self.compare_numerical_arg_1(neg, False)

    def test_add(self):
        self.compare_numerical_arg_2(add)

    def test_sub(self):
        self.compare_numerical_arg_2(sub)
        self.compare_numerical_arg_2(rsub)

    def test_mul(self):
        self.compare_numerical_arg_2(mul)

    def test_div(self):
        self.compare_numerical_arg_2(div)
        self.compare_numerical_arg_2(rdiv)

    def test_pow(self):
        x = Variable(np.random.rand(1))
        c = np.random.rand(1)
        y = pow(x, c)
        y.back_prop()

        # numerical
        eps = 1e-4
        x_minus = Variable(x.data - eps)
        x_plus = Variable(x.data + eps)
        y_minus = pow(x_minus, c)
        y_plus = pow(x_plus, c)
        expected = (y_plus.data - y_minus.data) / (2 * eps)

        flg = np.allclose(x.grad, expected)
        self.assertTrue(flg)

    def test_loop_topology(self):
        x = Variable(np.array(2.0))
        a = x*x
        y = a*a + a*a
        y.back_prop()

        flg = np.allclose(x.grad, np.array(64.0))
        self.assertTrue(flg)

    def test_sphere_func(self):
        def sphere(x, y):
            z = x**2 + y**2
            return z

        self.compare_numerical_arg_2(sphere)

    def test_matyas_func(self):
        def matyas(x, y):
            z = 0.26*(x**2 + y**2) - 0.48*x*y
            return z

        self.compare_numerical_arg_2(matyas)

    def test_goldstein_price_func(self):
        def goldstein_price(x, y):
            z = (1 + (x + y + 1)**2*(19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2))\
                * (30 + (2*x - 3*y)**2*(18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
            return z

        self.compare_numerical_arg_2(goldstein_price)


if __name__ == '__main__':
    unittest.main()
