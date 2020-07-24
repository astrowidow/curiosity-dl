import numpy as np
from kdnn.core import Function


class Square(Function):
    def forward(self, x):
        return x**2

    def backward(self, dl_dy):
        x, = self.input_variables
        dy_dx = 2*x
        dl_dx = dl_dy*dy_dx
        return dl_dx


def square(x):
    return Square()(x)


class Sin(Function):
    def forward(self, x):
        return np.sin(x)

    def backward(self, dl_dy):
        x, = self.input_variables
        dy_dx = cos(x)
        dl_dx = dl_dy*dy_dx
        return dl_dx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        return np.cos(x)

    def backward(self, dl_dy):
        x, = self.input_variables
        dy_dx = -sin(x)
        dl_dx = dl_dy * dy_dx
        return dl_dx


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, dl_dy):
        y = self.output_variable()
        dy_dx = 1 - y*y
        dl_dx = dl_dy*dy_dx
        return dl_dx


def tanh(x):
    return Tanh()(x)


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, dl_dy):
        x, = self.input_variables
        dy_dx = exp(x)
        dl_dx = dl_dy*dy_dx
        return dl_dx


def exp(x):
    return Exp()(x)
