# Artificial Neural Network From Scratch (NumPy)

This project demonstrates how to build and train an **Artificial Neural Network (ANN) from scratch using NumPy**, with a step-by-step explanation of **forward propagation**, **backpropagation**, and **gradient descent**.
It also includes a **comparison with scikit-learn’s `MLPClassifier`**, highlighting differences in abstraction, performance, and interpretability.

## Motivation

The goal of this project is to **demystify neural networks** by implementing them in plain Python and using the code itself as a learning tool.

This repository is directly related to the Medium article:

> **"Understanding Artificial Neural Networks by Implementing One from Scratch"**


## ⚙️ Technologies Used

* Python 3
* NumPy
* scikit-learn
* Matplotlib (optional, for visualization)

---

## 🔁 Neural Network From Scratch

### Model Overview

* Fully connected feedforward neural network
* Activation functions: Sigmoid / ReLU
* Loss function: Binary Cross-Entropy or MSE
* Optimization: Gradient Descent

### Key Methods

* `forward(X)`
* `backward(X, y)`
* `fit(X, y, epochs, learning_rate)`
* `predict(X)`
* `predict_proba(X)`

---



##  Comparison with Scikit-learn

The same dataset is trained using scikit-learn’s `MLPClassifier` to provide a fair comparison.


##  Medium Article

If you are interested in the full explanation with code walkthrough:

 **Medium Article:** *([link here](https://medium.com/@hjmartins88/artificial-neural-network-19ade83873d6))*


##  License

This project is open-source and available under the MIT License.
