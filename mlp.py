import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1, elu(x, alpha) + alpha)

class Sigmoid:
    def forward(self, x):
        self.x = x
        return sigmoid(x)

    def backward(self, x, dldy):
        dx = sigmoid_derivative(sigmoid(self.x)) * dldy
        return dx
    
class ReLu:
    def forward(self, x):
        self.x = x
        return relu(x)

    def backward(self, x, dldy):
        dx = relu_derivative(self.x) * dldy
        return dx

class Tanh:
    def forward(self, x):
        self.x = x
        return tanh(x)

    def backward(self, x, dldy):
        dx = tanh_derivative(tanh(self.x)) * dldy
        return dx

class LeakyReLU:
    def forward(self, x):
        self.x = x
        return leaky_relu(x)

    def backward(self, x, dldy):
        dx = leaky_relu_derivative(self.x) * dldy
        return dx

class ELU:
    def forward(self, x):
        self.x = x
        return elu(x)

    def backward(self, x, dldy):
        dx = elu_derivative(self.x) * dldy
        return dx

class Dropout:
    def __init__(self, rate=0.5):
        self.rate = rate

    def forward(self, x, train=True):
        if train:
            self.mask = np.random.binomial(1, 1 - self.rate, size=x.shape) / (1 - self.rate)
            return x * self.mask
        else:
            return x

    def backward(self, x, dldy):
        return dldy * self.mask

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
    def __init__(self, input_dim, output_dim, l2_reg=0.0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)  # He Initialization
        self.bias = np.zeros((1, output_dim))
        self.l2_reg = l2_reg

    def forward(self, x):
        self.x = x
        return np.dot(x, self.weight) + self.bias

    def backward(self, x, dldy):
        self.dweight = np.dot(self.x.T, dldy) + self.l2_reg * self.weight
        self.dbias = np.sum(dldy, axis=0, keepdims=True)
        return np.dot(dldy, self.weight.T)

# Feedforward Network Class
class FeedForwardNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x, train=True):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, train)
            else:
                x = layer.forward(x)
        return x

    def backward(self, dldy):
        for layer in reversed(self.layers):
            dldy = layer.backward(layer.x if hasattr(layer, 'x') else None, dldy)

def accuracy(y, t):
    predictions = np.argmax(y, axis=1)
    return np.mean(predictions == t)

# Training and Evaluating the MLP
def train_and_evaluate_manual_mlp(X_train, T_train, X_test, T_test, input_size, hidden_size, output_size, epochs, learning_rate, l2_reg=0.0, patience=10):
    layers = [
        Linear(input_size, hidden_size, l2_reg=l2_reg),
        ELU(),
        Dropout(rate=0.5),  # Add dropout with 50% rate
        Linear(hidden_size, output_size, l2_reg=l2_reg)
    ]

    mlp = FeedForwardNetwork(layers)
    ce_loss = CrossEntropy()

    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []

    best_test_loss = float('inf')
    early_stop_counter = 0
    best_model_weights = None
    best_model_biases = None

    for epoch in range(epochs):
        output = mlp.forward(X_train, train=True)
        loss = ce_loss.forward(output, T_train)
        dldy = ce_loss.backward(output, T_train)
        mlp.backward(dldy)

        for layer in mlp.layers:
            if isinstance(layer, Linear):
                layer.weight -= learning_rate * layer.dweight
                layer.bias -= learning_rate * layer.dbias

        train_acc = accuracy(mlp.forward(X_train, train=False), T_train)
        test_acc = accuracy(mlp.forward(X_test, train=False), T_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        train_loss_list.append(loss)
        test_loss_list.append(ce_loss.forward(mlp.forward(X_test, train=False), T_test))

        # Early stopping check
        if test_loss_list[-1] < best_test_loss:
            best_test_loss = test_loss_list[-1]
            early_stop_counter = 0
            # Save the best model weights
            best_model_weights = [layer.weight.copy() for layer in mlp.layers if isinstance(layer, Linear)]
            best_model_biases = [layer.bias.copy() for layer in mlp.layers if isinstance(layer, Linear)]
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss}, Train Accuracy: {train_acc}, Test Accuracy: {test_acc}')

    # Restore the best model weights
    if best_model_weights is not None and best_model_biases is not None:
        for i, layer in enumerate([layer for layer in mlp.layers if isinstance(layer, Linear)]):
            layer.weight = best_model_weights[i]
            layer.bias = best_model_biases[i]

    return mlp, train_acc_list, test_acc_list, train_loss_list, test_loss_list

# Training and Evaluating the Scikit-learn MLP
def train_and_evaluate_sklearn_mlp(X_train, T_train, X_test, T_test, hidden_layer_sizes, max_iter, learning_rate_init, alpha):
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, learning_rate_init=learning_rate_init, alpha=alpha, random_state=1)
    mlp.fit(X_train, T_train)
    
    train_predictions = mlp.predict(X_train)
    test_predictions = mlp.predict(X_test)
    
    train_accuracy = accuracy_score(T_train, train_predictions)
    test_accuracy = accuracy_score(T_test, test_predictions)
    
    return train_accuracy, test_accuracy, mlp