import numpy as np
from layers.conv2d import AcrobatConv2D
from layers.linear import RingmasterLinearLayer
from layers.flatten import ClownFlatten
from activations.relu import JugglerReLU
from activations.softmax import JugglerSoftmax
from utils.mnist_loader import load_mnist_images, load_mnist_labels, one_hot_encode
from network import CircusNet

# MaxPool layer (ручная реализация)
class ClownMaxPool2D:
    def __init__(self, size=2, stride=2):
        self.size = size
        self.stride = stride

    def forward(self, x):
        self.input = x
        N, C, H, W = x.shape
        out_h = (H - self.size) // self.stride + 1
        out_w = (W - self.size) // self.stride + 1
        out = np.zeros((N, C, out_h, out_w))
        self.max_mask = np.zeros_like(x)
        for i in range(out_h):
            for j in range(out_w):
                x_slice = x[:, :, i*self.stride:i*self.stride+self.size, j*self.stride:j*self.stride+self.size]
                out[:, :, i, j] = np.max(x_slice, axis=(2, 3))
                mask = (x_slice == out[:, :, i, j][:, :, None, None])
                self.max_mask[:, :, i*self.stride:i*self.stride+self.size, j*self.stride:j*self.stride+self.size] += mask
        return out

    def backward(self, grad_output):
        grad_input = np.zeros_like(self.input)
        N, C, H, W = self.input.shape
        out_h = (H - self.size) // self.stride + 1
        out_w = (W - self.size) // self.stride + 1
        for i in range(out_h):
            for j in range(out_w):
                grad = grad_output[:, :, i, j][:, :, None, None]
                grad_input[:, :, i*self.stride:i*self.stride+self.size, j*self.stride:j*self.stride+self.size] += grad * self.max_mask[:, :, i*self.stride:i*self.stride+self.size, j*self.stride:j*self.stride+self.size]
        return grad_input

# Load MNIST data
X = load_mnist_images('data/train-images-idx3-ubyte.gz')[:10000]
X = X.reshape(-1, 1, 28, 28)
y = load_mnist_labels('data/train-labels-idx1-ubyte.gz')[:10000]
y_oh = one_hot_encode(y, 10)

# Define model
model = CircusNet([
    AcrobatConv2D(1, 8, kernel_size=3, stride=1, padding='same'),
    JugglerReLU(),
    ClownMaxPool2D(),
    AcrobatConv2D(8, 8, kernel_size=3, stride=1, padding='same'),
    JugglerReLU(),
    ClownMaxPool2D(),
    ClownFlatten(),
    RingmasterLinearLayer(8 * 7 * 7, 128),
    JugglerReLU(),
    RingmasterLinearLayer(128, 10),
    JugglerSoftmax()
])

# Loss and gradient
def cross_entropy(pred, target):
    return -np.mean(np.sum(target * np.log(pred + 1e-8), axis=1))

def grad_cross_entropy(pred, target):
    return pred - target

# Training with minibatch
epochs = 20
batch_size = 32
lr = 0.02

for epoch in range(epochs):
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y_oh[indices]
    total_loss = 0
    correct = 0

    for i in range(0, len(X), batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        out = model.forward(X_batch)
        out_flat = out.reshape(out.shape[0], -1)
        loss = cross_entropy(out_flat, y_batch)
        grad = grad_cross_entropy(out_flat, y_batch)
        model.backward(grad, lr)
        total_loss += loss * len(X_batch)
        correct += np.sum(np.argmax(out_flat, axis=1) == np.argmax(y_batch, axis=1))

    avg_loss = total_loss / len(X)
    accuracy = correct / len(X)
    print(f"🎪 Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%")
