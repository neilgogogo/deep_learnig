import torch
from Module import Module


class ReLU(Module):
    """
    Implements Rectifier Linear Unit activation
    Only positive inputs are back-propagated
    """
    def forward(self, x):
        #saving x for the backpropagation
        self.x = x
        # return non-negative x
        return torch.max(torch.empty(self.x.size()).zero_(),x)

    def backward(self, eta):
        eta[self.x<=0]=0
        return eta

class Tanh(Module):
    """
    Implements tanh activation
    The derivative of tanh(x) is 1 - tanh(x) ** 2
    """
    def forward(self, x):
        ex = torch.exp(x)
        esx = torch.exp(-x)
        self.y = (ex - esx) / (ex + esx)
        return self.y

    def backward(self, eta):
        return eta*(1-self.y**2)
    
class Sigmoid(Module):
    """
    Implements sigmoid activation
    The derivative of Sigmoid(x) is Sigmoid(x)(1-Sigmoid(x))
    """
    def forward(self, x):
        self.x=x
        self.y=1/(1+torch.exp(-x))
        return self.y
    def backward(self, eta):
        return eta*(self.y*(1-self.y))
