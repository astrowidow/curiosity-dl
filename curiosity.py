import numpy as np


# Base classes
class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def back_prop(self):
        f = self.creator
        if f is not None:
            # y = f(x)
            # extracts the input variable 'x'
            variable_x = f.input_variable
            # dL/dx = dy/dx * dL/dy
            # dy/dx: df/dx
            # dL/dy: self.grad
            variable_x.grad = f.backward(self.grad)
            variable_x.back_prop()


class Function:
    def __call__(self, input_variable):
        self.input_variable = input_variable
        x = input_variable.data
        # y = f(x)
        y = self.forward(x)
        output_variable = Variable(y)
        output_variable.set_creator(self)
        return output_variable

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, dl_dy):
        # input: dL/dy
        # output: dL/dx = dy/dx * dL/dy
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
