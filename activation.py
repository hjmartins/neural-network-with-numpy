import numpy as np

def relu(x):
    return np.maximum(0, x)
# It means if x<=0 then f(x)=0, else f(x)=x. In the first case, when x<0 so the derivative of f(x) 
# with respect to x gives result f'(x)=0. In the second case, it's clear to compute f'(x)=1.
# founded in stack overflow: ReLU derivative in backpropagation
    # Derivative of ReLU
def relu_derivative(x):
    return (x > 0).astype(float)

# when i did the softmax i had overflow problems,
#  to solve it i subtracted the max from each element
# i found in this source: 
# https://medium.com/@akshat.dev/how-i-built-a-neural-network-from-scratch-using-only-python-and-numpy-a-step-by-step-journey-3345d67c6d2b
def softmax(x): 
    x_shift = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shift)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))