class CircusNet:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad, learning_rate):
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                if 'Linear' in layer.__class__.__name__ or 'Conv' in layer.__class__.__name__:
                    grad = layer.backward(grad, learning_rate)
                else:
                    grad = layer.backward(grad)
