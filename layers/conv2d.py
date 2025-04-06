
import numpy as np

class AcrobatConv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='valid'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.kernels = np.random.randn(out_channels, in_channels, *self.kernel_size) * 0.1
        self.bias = np.zeros((out_channels, 1))
        self.input = None
        self.grad_kernels = None
        self.grad_bias = None

    def pad_input(self, x):
        if self.padding == 'same':
            pad_h = ((x.shape[2] - 1) * self.stride + self.kernel_size[0] - x.shape[2]) // 2
            pad_w = ((x.shape[3] - 1) * self.stride + self.kernel_size[1] - x.shape[3]) // 2
            return np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        return x

    def forward(self, x):
        self.input = x
        x = self.pad_input(x)
        batch_size, in_c, h, w = x.shape
        kh, kw = self.kernel_size
        sh = sw = self.stride
        out_h = (h - kh) // sh + 1
        out_w = (w - kw) // sw + 1
        output = np.zeros((batch_size, self.out_channels, out_h, out_w))

        for i in range(out_h):
            for j in range(out_w):
                x_slice = x[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw]
                for k in range(self.out_channels):
                    output[:, k, i, j] = np.sum(x_slice * self.kernels[k], axis=(1,2,3)) + self.bias[k]
        return output

    def backward(self, grad_output, learning_rate):
        x = self.pad_input(self.input)
        batch_size, _, h, w = x.shape
        kh, kw = self.kernel_size
        sh = sw = self.stride
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]
        grad_input = np.zeros_like(x)
        self.grad_kernels = np.zeros_like(self.kernels)
        self.grad_bias = np.zeros_like(self.bias)

        for i in range(out_h):
            for j in range(out_w):
                x_slice = x[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw]
                for k in range(self.out_channels):
                    self.grad_kernels[k] += np.sum(x_slice * grad_output[:, k, i, j][:, None, None, None], axis=0)
                for n in range(batch_size):
                    grad_input[n, :, i*sh:i*sh+kh, j*sw:j*sw+kw] += np.sum(
                        self.kernels[:, :, :, :] * grad_output[n, :, i, j][:, None, None, None], axis=0
                    )
        self.grad_bias = np.sum(grad_output, axis=(0,2,3)).reshape(self.out_channels, 1)

        self.kernels -= learning_rate * self.grad_kernels / batch_size
        self.bias -= learning_rate * self.grad_bias / batch_size

        if self.padding == 'same':
            pad_h = ((self.input.shape[2] - 1) * self.stride + self.kernel_size[0] - self.input.shape[2]) // 2
            pad_w = ((self.input.shape[3] - 1) * self.stride + self.kernel_size[1] - self.input.shape[3]) // 2
            return grad_input[:, :, pad_h:-pad_h, pad_w:-pad_w]
        return grad_input
