
import numpy as np

class RingmasterLinearLayer:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros((1, output_dim))
        self.input = None
        self.grad_weights = None
        self.grad_bias = None

    def forward(self, input_data):
        self.input = input_data
        return np.dot(input_data, self.weights) + self.bias

    def backward(self, grad_output, learning_rate):
        batch_size = self.input.shape[0]
        self.grad_weights = np.dot(self.input.T, grad_output) / batch_size
        self.grad_bias = np.mean(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.weights.T)

        # Update
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias

        return grad_input
