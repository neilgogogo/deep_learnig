import math
import torch
from Module import Module
from Parameter import Parameter


class Linear(Module):
    """
    Implements linear layer, with or without bias
    Parameters: shape: (in_size, out_size)
    bias: whether there is bias in the Linear, defaut value is true
    initialization: uniform
    """

    def __init__(self, shape, bias=True, initialization='uniform', **kwargs):
        super().__init__()
        # using pytorch default weight initialization

        self.weights = Parameter(torch.Tensor(shape[0], shape[1]))
        if initialization == 'uniform':
            init_range = 1. / math.sqrt(shape[0])
            self.weights.data = self.weights.data.uniform_(-init_range, init_range)

        self.weights.grad = torch.zeros(shape)

        if bias:
            # default bias initialization
            self.bias = Parameter(torch.Tensor(shape[1]))
            self.bias.grad = torch.zeros(shape[1])
            if initialization == 'uniform':
                self.bias.data = self.bias.data.uniform_(-init_range, init_range)
        else:
            self.bias = None

    def forward(self, x):
        # Returns tensor of size N * out_features
        self.x = x
        output = x.matmul(self.weights.data)
        if self.bias is not None:
            output += self.bias.data
        return output

    def backward(self, grad_output):
        # grad_output: gradient from the lower layers
        # Returns tensor of size N * in_features, computes gradient wrt the weights

        self.weights.grad += torch.mm(self.x.t(), grad_output)
        #         torch.mm(a,b)
        grad_input = torch.mm(grad_output, self.weights.data.t())
        #         grad_input = grad_output.matmul(self.weights.data.t())

        if self.bias is not None:
            self.bias.grad += torch.einsum('ij->j', grad_output)
        return grad_input