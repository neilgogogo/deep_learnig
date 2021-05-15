class Module(object):
    """
    Base class to be inherited by other modules
    """
    def __init__(self):
        self.parameters = []

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError