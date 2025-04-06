# 🎪 Circus ANN Platform

This project implements an Artificial Neural Network (ANN) platform from scratch using NumPy.

## Features
- Linear (Fully Connected) Layers
- 1D & 2D Convolution Layers
- Activation Layers: ReLU, Sigmoid, Tanh, Softmax
- Manual Backpropagation

## Theme
All components are named with a circus theme to make the learning experience more fun!

## How to Run

### 3. Run Training Scripts

- Train on XOR (simple MLP):
  ```bash
  python train.py
  ```
- Train MLP on MNIST (784 -> 128 -> 10):
  ```bash
  python train_mnist.py
  ```
- Train CNN on MNIST (Conv + Pooling + FC):
  ```bash
  python train_mnist_conv.py
  ```


## Requirements
- Python 3.8+
- NumPy

## Structure
- `layers/`: Layer definitions
- `activations/`: Activation functions
- `utils/`: Helpers like loss functions, metrics
- `notebooks/`: Jupyter notebook demo
- `data/`: Sample data
