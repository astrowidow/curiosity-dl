import heapq
import weakref
import contextlib
import numpy as np


class Config:
    enable_back_prop = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config('enable_back_prop', False)


# Base classes
class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        # Generation is defined as negative number for 'heapq'
        self.generation = func.generation - 1

    def clear_grad(self):
        self.grad = None

    def back_prop(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        pushed_func_set = set()

        def push_func(func):
            if func not in pushed_func_set:
                heapq.heappush(funcs, func)
                # For example,
                # even when a 'Variable' is used as an argument to two or more functions,
                # the creator of the 'Variable' should be referenced only one time.
                pushed_func_set.add(func)

        push_func(self.creator)
        while funcs:
            f = heapq.heappop(funcs)
            # y = f(x)
            # extracts the input variables 'x'
            x_tuple = f.input_variables
            # y_variable = f.output_variable
            y_variable = f.output_variable()
            # dL/dx = dy/dx * dL/dy
            # dy/dx: df/dx
            # dL/dy: self.grad (It is assumed that there is only one output of 'f')
            x_grad_tuple = f.backward(y_variable.grad)
            if not isinstance(x_grad_tuple, tuple):
                x_grad_tuple = (x_grad_tuple,)
            for x_variable, x_grad in zip(x_tuple, x_grad_tuple):
                if x_variable.grad is None:
                    x_variable.grad = x_grad
                else:
                    # An implementation of 'x_variable.grad += x_grad' cannot be used.
                    # Because 'x_grad' has type of 'ndarray',
                    # 'x_grad' may be REFERENCE to 'y_grad'.
                    # This may cause overwriting 'y_grad'.
                    x_variable.grad = x_variable.grad + x_grad
                if x_variable.creator is not None:
                    push_func(x_variable.creator)

            if not retain_grad:
                y_variable.grad = None

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        return Variable(self.data.T)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Function:
    def __call__(self, *input_variables):
        input_variables = [as_variable(input_val) for input_val in input_variables]
        x_tuple = (input_val.data for input_val in input_variables)
        # y = f(x)
        # 'x' is 'tuple' of 'ndarray' because of the case that there are multiple inputs
        # 'y' is assumed that its type is 'ndarray'
        y = self.forward(*x_tuple)
        # There is the case that 'y' is returned as scalar
        # 'as_array' turns type 'scalar' into type 'ndarray'
        output_variable = Variable(as_array(y))
        if Config.enable_back_prop:
            self.input_variables = input_variables
            self.generation = min([input_val.generation for input_val in input_variables])
            output_variable.set_creator(self)
            # Because 'Function.output_variable' and 'Variable.creator' form a circular reference,
            # it's needed to be referenced weakly.
            # self.output_variable = output_variable
            self.output_variable = weakref.ref(output_variable)
        return output_variable

    def __lt__(self, other):
        return self.generation < other.generation

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, dl_dy):
        # input: dL/dy
        # output: dL/dx = dy/dx * dL/dy
        raise NotImplementedError()


# Operator functions
class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, dl_dy):
        return -dl_dy


def neg(x):
    return Neg()(x)


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, dl_dy):
        dl_dx0 = dl_dy
        dl_dx1 = dl_dy
        return dl_dx0, dl_dx1


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, dl_dy):
        dl_dx0 = dl_dy
        dl_dx1 = -dl_dy
        return dl_dx0, dl_dx1


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, dl_dy):
        dl_dx0 = dl_dy*self.input_variables[1].data
        dl_dx1 = dl_dy*self.input_variables[0].data
        return dl_dx0, dl_dx1


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Div(Function):
    def forward(self, x0, x1):
        y = x0/x1
        return y

    def backward(self, dl_dy):
        x0 = self.input_variables[0].data
        x1 = self.input_variables[1].data
        dl_dx0 = dl_dy / x1
        dl_dx1 = dl_dy * (-x0/x1**2)
        return dl_dx0, dl_dx1


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, dl_dy):
        x = self.input_variables[0].data
        c = self.c
        dl_dx = c*x**(c - 1)*dl_dy
        return dl_dx


def pow(x, c):
    return Pow(c)(x)


def setup_variable():
    Variable.__neg__ = neg
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
