from Linear import Linear


class SGD:
    def __init__(self, layers, learning_rate):
        self.layers = layers
        self.learning_rate = learning_rate

    def step(self):

        for layer in self.layers:
            # if the Layer is Linear, update the weight and bias
            if isinstance(layer, Linear):
                # update the weight and bias
                layer.weights.data = layer.weights.data - self.learning_rate * layer.weights.grad
                layer.bias.data = layer.bias.data - self.learning_rate * layer.bias.grad