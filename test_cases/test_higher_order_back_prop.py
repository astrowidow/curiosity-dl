if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from kdnn import Variable
import kdnn.functions as kf
import unittest


def _numerical_diff_arg_1(f, x, eps=1e-4):
    x_minus = Variable(np.array(x.data - eps))
    x_plus = Variable(np.array(x.data + eps))
    y_minus = f(x_minus)
    y_plus = f(x_plus)
    return (y_plus - y_minus)/(2*eps)


def _numerical_double_diff_arg_1(f, x, eps=5e-4):
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
        x = Variable(1*np.random.rand(1) + 1)
        y = f(x)

        # first order differential
        y.back_prop(retain_grad, create_graph)
        xd = x.grad

        # second order differential
        x.clear_grad()
        try:
            xd.back_prop(retain_grad, create_graph)
        except AttributeError as e:
            self.assertFalse(create_graph)
            return
        else:
            self.assertTrue(create_graph)

        expected = _numerical_double_diff_arg_1(f, x)
        diff_norm = max(np.abs(expected.data - np.zeros_like(expected.data)))
        flg = diff_norm < 5e-4
        if flg:
            self.assertEqual(x.grad, None)
            return
        flg = np.allclose(x.grad.data, expected.data)
        self.assertTrue(flg)
        if retain_grad:
            flg = np.allclose(y.grad.data, np.ones_like(y.data))
            self.assertTrue(flg)
        else:
            self.assertEqual(y.grad, None)

        # # third order differential
        # xdd = x.grad
        # x.clear_grad()
        # try:
        #     xdd.back_prop(retain_grad, create_graph)
        # except AttributeError as e:
        #     self.assertFalse(create_graph)
        #     return
        # else:
        #     self.assertTrue(create_graph)
        # expected = _numerical_triple_diff_arg_1(f, x)
        # diff_norm = max(np.abs(expected.data - np.zeros_like(expected.data)))
        # flg = diff_norm < 1e-5
        # if flg:
        #     self.assertEqual(x.grad, None)
        #     return
        #
        # diff_norm = max(np.abs(x.grad.data - expected.data))
        # flg = diff_norm < 1e-5
        # self.assertTrue(flg)

    def test_polynomial(self):
        def polynomial(x):
            y = x**4 - 2*x**2
            return y
        self.compare_numerical_arg_1(polynomial, True, True)
        self.compare_numerical_arg_1(polynomial, False, True)
        self.compare_numerical_arg_1(polynomial, True, False)
        self.compare_numerical_arg_1(polynomial, False, False)

    def test_exp(self):
        self.compare_numerical_arg_1(kf.exp, True, True)
        self.compare_numerical_arg_1(kf.exp, False, True)
        self.compare_numerical_arg_1(kf.exp, True, False)
        self.compare_numerical_arg_1(kf.exp, False, False)

    def test_squared(self):
        self.compare_numerical_arg_1(kf.square, True, True)
        self.compare_numerical_arg_1(kf.square, False, True)
        self.compare_numerical_arg_1(kf.square, True, False)
        self.compare_numerical_arg_1(kf.square, False, False)

    def test_sin(self):
        self.compare_numerical_arg_1(kf.sin, True, True)
        self.compare_numerical_arg_1(kf.sin, False, True)
        self.compare_numerical_arg_1(kf.sin, True, False)
        self.compare_numerical_arg_1(kf.sin, False, False)

    def test_cos(self):
        self.compare_numerical_arg_1(kf.cos, True, True)
        self.compare_numerical_arg_1(kf.cos, False, True)
        self.compare_numerical_arg_1(kf.cos, True, False)
        self.compare_numerical_arg_1(kf.cos, False, False)

    def test_tanh(self):
        x = Variable(1 * np.random.rand(1) + 1)
        y = kf.tanh(x)
        xdd_analytical = -2*y*(1 - y*y)

        # first order differential
        y.back_prop(create_graph=True)
        xd = x.grad
        # second order differential
        x.clear_grad()
        xd.back_prop(create_graph=True)
        xdd = x.grad
        flg = np.allclose(xdd.data, xdd_analytical.data)
        self.assertTrue(flg)

    def test_sin_third_order_diff(self):
        x = Variable(1 * np.random.rand(1) + 1)
        y = kf.sin(x)

        # first order differential
        y.back_prop(create_graph=True)
        xd = x.grad
        # second order differential
        x.clear_grad()
        xd.back_prop(create_graph=True)
        xdd = x.grad
        # third order differential
        x.clear_grad()
        xdd.back_prop()
        xddd = x.grad

        # analytical solution
        xddd_analytical = -kf.cos(x)
        flg = np.allclose(xddd.data, xddd_analytical.data)
        self.assertTrue(flg)

    def test_diff_including_diff(self):
        def answer_func(_x):
            return 24*_x**2 + 2*_x
        x = Variable(np.array(np.random.rand(1)))
        y = x**2
        y.back_prop(create_graph=True)
        xd = x.grad
        x.clear_grad()
        z = xd**3 + y
        z.back_prop()
        zd = x.grad
        answer = answer_func(x)
        flg = np.allclose(zd.data, answer.data)
        self.assertTrue(flg)


if __name__ == '__main__':
    unittest.main()
