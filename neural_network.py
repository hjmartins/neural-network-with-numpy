import numpy as np
from activation import relu, softmax, relu_derivative, sigmoid
from losses import categorial_cross_entropy, binary_cross_entropy
""" | Nome | Significado              |
| ---- | ------------------------ |
| W1   | Pesos da camada 1        |
| Z1   | Pré-ativação da camada 1 |
| A1   | Saída da camada 1        |
| W2   | Pesos da camada 2        |
| Z2   | Pré-ativação da camada 2 |
| A2   | Saída final              |
| b1   | Bias da camada 1         |
| b2   | Bias da camada 2         |
| lr   | Taxa de aprendizado      |
"""
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, lr=0.01):

        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, 1))
        self.lr = lr
 
#    def forward(self, X):
#        self.z1 = X.dot(self.W1)    # old code without bias
#        self.a1 = relu(self.z1)  
#        self.z2 = self.a1.dot(self.W2)  
#        self.a2 = softmax(self.z2)  
#        return(self.a2)

    def forward(self, X):
        self.z1 = X.dot(self.W1) + self.b1      # adding bias to the first layer
        self.a1 = relu(self.z1)
        self.z2 = self.a1.dot(self.W2) + self.b2   # adding bias to the second layer 
        self.a2 = sigmoid(self.z2)

        return self.a2
    
    def backward(self, X, y): #backpropagation
        m = X.shape[0]

        dz2 = self.a2 - y.reshape(-1, 1)


        #dz2 = self.a2 - y
        dW2 = (self.a1.T @ dz2) / m
        db2 = dz2.mean(axis=0, keepdims=True)

        dz1 = (dz2 @ self.W2.T) * relu_derivative(self.z1)
        dW1 = X.T @ dz1 / m
        db1 = dz1.mean(axis=0, keepdims=True)
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    def fit(self, X, y, epochs=1000, batch_size=32):
        losses = []
        for epoch in range(epochs):
            indices = np.random.permutation(len(X)) #or X.shape[0]
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            for i in range(0, len(X), batch_size): # or X.shape[0] 
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                y_hat = self.forward(X_batch)
                #self.forward(X_batch) # foward pass for categorical cross entropy and softmax
                self.backward(X_batch, y_batch)
            #y_pred = self.forward(X_batch)
            loss = binary_cross_entropy(y, self.forward(X))
            losses.append(loss)
            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
        return losses
    
    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
