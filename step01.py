import numpy as np


class Variable:
    def __init__(self, data):

        assert isinstance(
            data, np.ndarray), f"type error {type(data)} is not supported , data's type must be np.ndarray"
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    # def backward(self):

    #     f = self.creator
    #     if f is not None:
    #         x = f.input
    #         x.grad = f.backward(self.grad)
    #         x.backward()

    def backward(self):

        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    def __call__(self, input):  # input type : Variable
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input
        self.output = output

        return output  # output type : Variable

    def forward(self, data):
        raise NotImplementedError()

    def backward(self, dy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, data):
        return data**2

    def backward(self, dy):
        x = self.input.data
        print(f"x : {x} , dy = {dy} , return : {2*x*dy}")
        return 2*x*dy


class Exp(Function):
    def forward(self, data):
        return np.exp(data)

    def backward(self, dy):
        x = self.input.data
        print(f"x : {x} , dy = {dy} , return : {np.exp(x)*dy}")
        return np.exp(x)*dy


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / 2*eps


def square(x):
    f = Square()
    return f(x)


def exp(x):
    f = Exp()
    return f(x)


x = Variable(np.array(2))
