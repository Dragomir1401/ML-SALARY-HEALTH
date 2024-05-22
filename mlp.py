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
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


# Linear Layer Class
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
        return dldy

    def update(self, *args, **kwargs):
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.update(*args, **kwargs)

class Linear:
    def __init__(self, input_dim, output_dim, l2_reg=0.01):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)  # He Initialization
        self.bias = np.zeros((1, output_dim))
        self.l2_reg = l2_reg
        self.dweight = np.zeros_like(self.weight)
        self.dbias = np.zeros_like(self.bias)

    def forward(self, x):
        self.x = x
        return np.dot(x, self.weight) + self.bias

    def backward(self, x, dldy):
        self.dweight = np.dot(self.x.T, dldy) + self.l2_reg * self.weight
        self.dbias = np.sum(dldy, axis=0, keepdims=True)
        return np.dot(dldy, self.weight.T)

    def update(self, mode='SGD', lr=0.001):
        if mode == 'SGD':
            self.weight -= lr * self.dweight
            self.bias -= lr * self.dbias
        else:
            raise ValueError('mode should be SGD, not ' + str(mode))

def accuracy(y, t):
    predictions = np.argmax(y, axis=1)
    return np.mean(predictions == t)

class SGDOptimizer:
    def __init__(self, layers, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.layers = [layer for layer in layers if isinstance(layer, Linear)]

    def update(self):
        for layer in self.layers:
            layer.weight -= self.learning_rate * layer.dweight
            layer.bias -= self.learning_rate * layer.dbias

# Training and Evaluating the MLP
def train_and_evaluate_manual_mlp(X_train, T_train, X_test, T_test, input_size, hidden_size, output_size, epochs, learning_rate, l2_reg=0.0, batch_size=32):
    layers = [
        Linear(input_size, hidden_size, l2_reg=l2_reg),
        ReLu(),
        Dropout(rate=0.5),
        Linear(hidden_size, output_size, l2_reg=l2_reg)
    ]

    net = FeedForwardNetwork(layers)
    cost_function = CrossEntropy()
    optimizer_args = {'mode': 'SGD', 'lr': learning_rate}

    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []

    for epoch in range(epochs):
        for b_no, idx in enumerate(range(0, len(X_train), batch_size)):
            x_batch = X_train[idx:idx + batch_size]
            t_batch = T_train[idx:idx + batch_size]

            # Forward pass
            y = net.forward(x_batch, train=True)
            loss = cost_function.forward(y, t_batch)

            # Backward pass (compute gradients)
            dy = cost_function.backward(y, t_batch)
            net.backward(dy)

            # Update parameters with optimizer
            net.update(**optimizer_args)

            print(f'\rEpoch {epoch + 1:02d} | Batch {b_no:03d} | Train NLL: {loss:.6f} | Train Acc: {accuracy(y, t_batch) * 100:.2f}% ', end='')

        # Evaluate on training and test sets
        train_output = net.forward(X_train, train=False)
        test_output = net.forward(X_test, train=False)
        train_acc = accuracy(train_output, T_train)
        test_acc = accuracy(test_output, T_test)
        train_loss = cost_function.forward(train_output, T_train)
        test_loss = cost_function.forward(test_output, T_test)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        print(f'| Test NLL: {test_loss:.6f} | Test Acc: {test_acc * 100:.2f}%')

    return net, train_acc_list, test_acc_list, train_loss_list, test_loss_list

# Training and Evaluating the Scikit-learn MLP
def train_and_evaluate_sklearn_mlp(X_train, T_train, X_test, T_test, hidden_layer_sizes, max_iter, learning_rate_init, alpha):
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, learning_rate_init=learning_rate_init, alpha=alpha, random_state=1)
    mlp.fit(X_train, T_train)
    
    train_predictions = mlp.predict(X_train)
    test_predictions = mlp.predict(X_test)
    
    train_accuracy = accuracy_score(T_train, train_predictions)
    test_accuracy = accuracy_score(T_test, test_predictions)
    
    return train_accuracy, test_accuracy, mlp
