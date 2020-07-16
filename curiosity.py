import numpy as np


# Base classes
class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None


class Function:
    def __call__(self, input_variable):
        self.input_variable = input_variable
        x = input_variable.data
        y = self.forward(x)
        output_variable = Variable(y)
        return output_variable

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, dl_dy):
        raise NotImplementedError()


# Element functions
class Squared(Function):
    def forward(self, x):
        return x**2

    def backward(self, dl_dy):
        x = self.input_variable.data
        dy_dx = 2*x
        dl_dx = dl_dy*dy_dx
        return dl_dx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, dl_dy):
        x = self.input_variable.data
        dy_dx = np.exp(x)
        dl_dx = dl_dy*dy_dx
        return dl_dx
