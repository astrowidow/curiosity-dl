import numpy as np


# Base classes
class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def clear_grad(self):
        self.grad = None

    def back_prop(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        f = self.creator
        if f is not None:
            # y = f(x)
            # extracts the input variables 'x'
            x_tuple = f.input_variables
            # dL/dx = dy/dx * dL/dy
            # dy/dx: df/dx
            # dL/dy: self.grad (It is assumed that there is only one output of 'f')
            x_grad_tuple = f.backward(self.grad)
            if not isinstance(x_grad_tuple, tuple):
                x_grad_tuple = (x_grad_tuple,)
            for x_variable, x_grad in zip(x_tuple, x_grad_tuple):
                if x_variable.grad is None:
                    x_variable.grad = x_grad
                else:
                    # An implementation of 'x_variable.grad += x_grad' cannot be used.
                    # Because 'x_grad' has type of 'ndarray',
                    # 'x_grad' may be REFERENCE to 'y_grad'.
                    x_variable.grad = x_variable.grad + x_grad
                x_variable.back_prop()


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *input_variables):
        # allows multiple arguments
        self.input_variables = input_variables
        x_tuple = (input_val.data for input_val in input_variables)
        # y = f(x)
        # 'x' is 'tuple' of 'ndarray' because of the case that there are multiple inputs
        # 'y' is assumed that its type is 'ndarray'
        y = self.forward(*x_tuple)
        # There is the case that 'y' is returned as scalar
        # 'as_array' turns type 'scalar' into type 'ndarray'
        output_variable = Variable(as_array(y))
        output_variable.set_creator(self)
        return output_variable

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, dl_dy):
        # input: dL/dy
        # output: dL/dx = dy/dx * dL/dy
        raise NotImplementedError()


# Operator functions
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, dl_dy):
        dl_dx0 = dl_dy
        dl_dx1 = dl_dy
        return dl_dx0, dl_dx1


def add(x0, x1):
    return Add()(x0, x1)


# Element functions
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
