from Activation import ReLU, Tanh, Sigmoid
from Linear import Linear
from Module import Module


class Sequential(Module):
    """
    Implements sequential layer to combine multiple modules given in *args
    # Usage example: Sequential(layer1, layer2, layer3)
    """

    def __init__(self, layer_configures, model_name):
        self.layers = []
        self.model_name = model_name
        for config in layer_configures:
            self.layers.append(self.createLayer(config))


    def createLayer(self, config):
        t = config['type']
        if t == 'Linear':
            layer = Linear(**config)
        elif t == 'ReLU':
            layer = ReLU()
        elif t == 'softmax':
            layer = Softmax()
        elif t == 'Tanh':
            layer = Tanh()
        elif t == 'Sigmoid':
            layer = Sigmoid()
        else:
            raise TypeError
        return layer

    def __call__(self, x):
        return self.forward(x)

    def zero_grad(self):
        for layer in self.layers:
            # if the Layer is Linear, update the gradient of weight and bias to zeros
            if isinstance(layer, Linear):
                # update the gradient of weight and bias to zeros
                layer.weights.grad.zero_()
                layer.bias.grad.zero_()

    def forward(self, x):

        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, eta):

        for layer in reversed(self.layers):
            eta = layer.backward(eta)
        return eta