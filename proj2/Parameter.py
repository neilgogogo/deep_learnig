class Parameter(object):
    def __init__(self, data):
        # data is used to store weight and bias
        self.data = data
        # grad is used to store gradient of weight and bias
        self.grad = None