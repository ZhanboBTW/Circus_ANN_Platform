
import numpy as np

class JugglerSoftmax:
    def forward(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.out = e_x / np.sum(e_x, axis=1, keepdims=True)
        return self.out

    def backward(self, grad_output):
        # Using simplified version: gradient already handled in loss
        return grad_output
