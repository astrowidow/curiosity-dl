import numpy as np


# Base classes
class Variable:
    def __init__(self, data):
        self.data = data


class Function:
    def __call__(self, input_variable):
        x = input_variable.data
        y = self.forward(x)
        output_variable = Variable(y)
        return output_variable

    def forward(self, x):
        raise NotImplementedError()


# Element functions
class Squared(Function):
    def forward(self, x):
        return x**2


class Exp(Function):
    def forward(self, x):
        return np.exp(x)
