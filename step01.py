import numpy as np
class Variable:
    def __init__(self , data):
        self.data = data
        
class Function:
    def __call__(self, input):
        x = input.data
        y= self.forward(x)
        output = Variable(y)
        return output
    def forward(self , data):
        raise NotImplementedError()

class Square(Function):
    def forward(self , data):
        return data**2

class Exp(Function):
    def forward(self,data):
        return np.exp(data)

x = Variable(np.array(10))
f = Square()
f2 = Exp()
y = f2(x)

print(x.data)
print(y.data)
    