import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, x, dldy):
        dx = np.array(dldy, copy=True)
        dx[self.x <= 0] = 0
        return dx

class CrossEntropy:
    def forward(self, y, t):
        self.y = y
        self.t = t
        m = t.shape[0]
        p = softmax(y)
        log_likelihood = -np.log(p[range(m), t])
        loss = np.sum(log_likelihood) / m
        return loss

    def backward(self, y, t):
        m = t.shape[0]
        grad = softmax(y)
        grad[range(m), t] -= 1
        grad = grad / m
        return grad

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

# Linear Layer Class
class Linear:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)  # He Initialization
        self.bias = np.zeros((1, output_dim))

    def forward(self, x):
        self.x = x
        return np.dot(x, self.weight) + self.bias

    def backward(self, x, dldy):
        self.dweight = np.dot(self.x.T, dldy)
        self.dbias = np.sum(dldy, axis=0, keepdims=True)
        return np.dot(dldy, self.weight.T)

# Feedforward Network Class
class FeedForwardNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dldy):
        for layer in reversed(self.layers):
            dldy = layer.backward(layer.x, dldy)

def accuracy(y, t):
    predictions = np.argmax(y, axis=1)
    return np.mean(predictions == t)

# Training and Evaluating the MLP
def train_and_evaluate_manual_mlp(X_train, T_train, X_test, T_test, input_size, hidden_size, output_size, epochs, learning_rate):
    layers = [
        Linear(input_size, hidden_size),
        ReLU(),
        Linear(hidden_size, output_size)
    ]

    mlp = FeedForwardNetwork(layers)
    ce_loss = CrossEntropy()

    for epoch in range(epochs):
        output = mlp.forward(X_train)
        loss = ce_loss.forward(output, T_train)
        dldy = ce_loss.backward(output, T_train)
        mlp.backward(dldy)

        for layer in mlp.layers:
            if isinstance(layer, Linear):
                layer.weight -= learning_rate * layer.dweight
                layer.bias -= learning_rate * layer.dbias

        if (epoch + 1) % 100 == 0:
            train_acc = accuracy(output, T_train)
            print(f'Epoch {epoch + 1}, Loss: {loss}, Train Accuracy: {train_acc}')

    train_acc = accuracy(mlp.forward(X_train), T_train)
    test_acc = accuracy(mlp.forward(X_test), T_test)
    return train_acc, test_acc


# Training and Evaluating the Scikit-learn MLP
def train_and_evaluate_sklearn_mlp(X_train, T_train, X_test, T_test, hidden_layer_sizes, max_iter, learning_rate_init, alpha):
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, learning_rate_init=learning_rate_init, alpha=alpha, random_state=1)
    mlp.fit(X_train, T_train)
    
    train_predictions = mlp.predict(X_train)
    test_predictions = mlp.predict(X_test)
    
    train_accuracy = accuracy_score(T_train, train_predictions)
    test_accuracy = accuracy_score(T_test, test_predictions)
    
    return train_accuracy, test_accuracy, mlp