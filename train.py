
import numpy as np
from layers.linear import RingmasterLinearLayer
from activations.relu import JugglerReLU, JugglerSigmoid
from network import CircusNet

# Create synthetic data: XOR pattern
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Define model
model = CircusNet([
    RingmasterLinearLayer(2, 4),
    JugglerReLU(),
    RingmasterLinearLayer(4, 1),
    JugglerSigmoid()
])

def binary_cross_entropy(pred, target):
    return -np.mean(target * np.log(pred + 1e-8) + (1 - target) * np.log(1 - pred + 1e-8))

def grad_bce(pred, target):
    return (pred - target) / (pred * (1 - pred) + 1e-8)

# Train
epochs = 10000
lr = 0.02
for epoch in range(epochs):
    pred = model.forward(X)
    loss = binary_cross_entropy(pred, y)
    grad = grad_bce(pred, y)
    model.backward(grad, lr)
    if epoch % 1000 == 0:
        print(f"🎪 Epoch {epoch}, Loss: {loss:.4f}")

# Final predictions
print("🎯 Final predictions:")
print(model.forward(X))
