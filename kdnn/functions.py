import numpy as np
from kdnn.core_simplified import Function


class Square(Function):
    def forward(self, x):
        return x**2

    def backward(self, dl_dy):
        x = self.input_variables[0].data
        dy_dx = 2*x
        dl_dx = dl_dy*dy_dx
        return dl_dx


def square(x):
    return Square()(x)


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, dl_dy):
        x = self.input_variables[0].data
        dy_dx = np.exp(x)
        dl_dx = dl_dy*dy_dx
        return dl_dx


def exp(x):
    return Exp()(x)
