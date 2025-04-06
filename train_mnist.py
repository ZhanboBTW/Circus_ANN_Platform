import numpy as np
from layers.linear import RingmasterLinearLayer
from activations.relu import JugglerReLU
from activations.softmax import JugglerSoftmax
from utils.mnist_loader import load_mnist_images, load_mnist_labels, one_hot_encode
from network import CircusNet

# Load MNIST
X_train = load_mnist_images('data/train-images-idx3-ubyte.gz')[:20000]
y_train = load_mnist_labels('data/train-labels-idx1-ubyte.gz')[:20000]
y_train_oh = one_hot_encode(y_train)

# Define model with Softmax
model = CircusNet([
    RingmasterLinearLayer(784, 128),
    JugglerReLU(),
    RingmasterLinearLayer(128, 10),
    JugglerSoftmax()
])

def cross_entropy(pred, target):
    return -np.mean(np.sum(target * np.log(pred + 1e-8), axis=1))

def grad_cross_entropy(pred, target):
    return pred - target  # For softmax + CE combo

# Training
epochs = 501
lr = 0.02
for epoch in range(epochs):
    pred = model.forward(X_train)
    loss = cross_entropy(pred, y_train_oh)
    grad = grad_cross_entropy(pred, y_train_oh)
    model.backward(grad, lr)
    acc = (np.argmax(pred, axis=1) == y_train).mean()
    if epoch % 50 == 0:
        print(f"🎪 Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc*100:.2f}%")
