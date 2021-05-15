import torch

from Module import Module


class MSELoss(Module):

    def __init__(self):
        super().__init__()

    def __call__(self, pred_, target):
        return self.forward(pred_, target)

    def forward(self, pred_, target):
        return torch.mean((pred_ - target) ** 2).item()

    def backward(self, pred_, target):
        # Computing the gradient of the loss with respect to input

        return 2 * (pred_ - target) / target.size(0)