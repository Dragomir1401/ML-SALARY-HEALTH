import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Activation Functions
def sigmoid(x):
    """Apply the sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Compute the derivative of the sigmoid function."""
    return x * (1 - x)

def relu(x):
    """Apply the ReLU activation function."""
    return np.maximum(0, x)

def relu_derivative(x):
    """Compute the derivative of the ReLU function."""
    return np.where(x > 0, 1, 0)

def tanh(x):
    """Apply the tanh activation function."""
    return np.tanh(x)

def tanh_derivative(x):
    """Compute the derivative of the tanh function."""
    return 1 - np.tanh(x)**2

def leaky_relu(x, alpha=0.01):
    """Apply the leaky ReLU activation function."""
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    """Compute the derivative of the leaky ReLU function."""
    return np.where(x > 0, 1, alpha)

def elu(x, alpha=1.0):
    """Apply the ELU activation function."""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    """Compute the derivative of the ELU function."""
    return np.where(x > 0, 1, elu(x, alpha) + alpha)

# Activation Function Classes
class Sigmoid:
    def forward(self, x):
        """Forward pass through the sigmoid activation function."""
        self.x = x
        return sigmoid(x)

    def backward(self, x, dldy):
        """Backward pass through the sigmoid activation function."""
        dx = sigmoid_derivative(sigmoid(self.x)) * dldy
        return dx

class ReLu:
    def forward(self, x):
        """Forward pass through the ReLU activation function."""
        self.x = x
        return relu(x)

    def backward(self, x, dldy):
        """Backward pass through the ReLU activation function."""
        dx = relu_derivative(self.x) * dldy
        return dx

class Tanh:
    def forward(self, x):
        """Forward pass through the tanh activation function."""
        self.x = x
        return tanh(x)

    def backward(self, x, dldy):
        """Backward pass through the tanh activation function."""
        dx = tanh_derivative(tanh(self.x)) * dldy
        return dx

class LeakyReLU:
    def forward(self, x):
        """Forward pass through the leaky ReLU activation function."""
        self.x = x
        return leaky_relu(x)

    def backward(self, x, dldy):
        """Backward pass through the leaky ReLU activation function."""
        dx = leaky_relu_derivative(self.x) * dldy
        return dx

class ELU:
    def forward(self, x):
        """Forward pass through the ELU activation function."""
        self.x = x
        return elu(x)

    def backward(self, x, dldy):
        """Backward pass through the ELU activation function."""
        dx = elu_derivative(self.x) * dldy
        return dx

# Dropout Layer Class
class Dropout:
    def __init__(self, rate=0.5):
        """Initialize the dropout layer with a dropout rate."""
        self.rate = rate

    def forward(self, x, train=True):
        """Forward pass through the dropout layer."""
        if train:
            # Apply dropout during training
            self.mask = np.random.binomial(1, 1 - self.rate, size=x.shape) / (1 - self.rate)
            return x * self.mask
        else:
            # Do not apply dropout during testing
            return x

    def backward(self, x, dldy):
        """Backward pass through the dropout layer."""
        return dldy * self.mask

# Cross-Entropy Loss Class
class CrossEntropy:
    def forward(self, y, t):
        """Forward pass to compute the cross-entropy loss."""
        self.y = y
        self.t = t
        m = t.shape[0]
        p = softmax(y)
        log_likelihood = -np.log(p[range(m), t])
        loss = np.sum(log_likelihood) / m
        return loss

    def backward(self, y, t):
        """Backward pass to compute the gradient of the loss."""
        m = t.shape[0]
        grad = softmax(y)
        grad[range(m), t] -= 1
        grad = grad / m
        return grad

def softmax(x):
    """Compute the softmax of each row of the input x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

# Linear Layer Class
class Linear:
    def __init__(self, input_dim, output_dim, l2_reg=0.01):
        """Initialize a linear layer with input and output dimensions."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
        self.bias = np.zeros((1, output_dim))
        self.l2_reg = l2_reg
        self.dweight = np.zeros_like(self.weight)
        self.dbias = np.zeros_like(self.bias)

    def forward(self, x):
        """Forward pass through the linear layer."""
        self.x = x
        return np.dot(x, self.weight) + self.bias

    def backward(self, x, dldy):
        """Backward pass through the linear layer."""
        self.dweight = np.dot(self.x.T, dldy) + self.l2_reg * self.weight
        self.dbias = np.sum(dldy, axis=0, keepdims=True)
        return np.dot(dldy, self.weight.T)

    def update(self, mode='SGD', lr=0.001):
        """Update the weights and biases using SGD."""
        if mode == 'SGD':
            self.weight -= lr * self.dweight
            self.bias -= lr * self.dbias
        else:
            raise ValueError('mode should be SGD, not ' + str(mode))

# Feed-Forward Network Class
class FeedForwardNetwork:
    def __init__(self, layers):
        """Initialize the feed-forward network with the given layers."""
        self.layers = layers

    def forward(self, x, train=True):
        """Forward pass through the network."""
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, train)
            else:
                x = layer.forward(x)
        return x

    def backward(self, dldy):
        """Backward pass through the network."""
        for layer in reversed(self.layers):
            dldy = layer.backward(layer.x if hasattr(layer, 'x') else None, dldy)
        return dldy

    def update(self, *args, **kwargs):
        """Update the parameters of the network."""
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.update(*args, **kwargs)

# Accuracy Function
def accuracy(y, t):
    """Compute the accuracy of predictions y against true labels t."""
    predictions = np.argmax(y, axis=1)
    return np.mean(predictions == t)

# SGD Optimizer Class
class SGDOptimizer:
    def __init__(self, layers, learning_rate=0.001):
        """Initialize the SGD optimizer with the given learning rate."""
        self.learning_rate = learning_rate
        self.layers = [layer for layer in layers if isinstance(layer, Linear)]

    def update(self):
        """Update the weights and biases using SGD."""
        for layer in self.layers:
            layer.weight -= self.learning_rate * layer.dweight
            layer.bias -= self.learning_rate * layer.dbias

# Adam Optimizer Class
class AdamOptimizer:
    def __init__(self, layers, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Initialize the Adam optimizer with the given parameters."""
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.layers = [layer for layer in layers if isinstance(layer, Linear)]
        self.t = 0
        self.m = [np.zeros_like(layer.weight) for layer in self.layers]
        self.v = [np.zeros_like(layer.weight) for layer in self.layers]
        self.m_bias = [np.zeros_like(layer.bias) for layer in self.layers]
        self.v_bias = [np.zeros_like(layer.bias) for layer in self.layers]

    def update(self):
        """Update the weights and biases using the Adam optimizer."""
        self.t += 1
        for i, layer in enumerate(self.layers):
            # Compute biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * layer.dweight
            # Compute biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (layer.dweight ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update weights
            layer.weight -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            # Compute biased first moment estimate for bias
            self.m_bias[i] = self.beta1 * self.m_bias[i] + (1 - self.beta1) * layer.dbias
            # Compute biased second moment estimate for bias
            self.v_bias[i] = self.beta2 * self.v_bias[i] + (1 - self.beta2) * (layer.dbias ** 2)

            # Compute bias-corrected first moment estimate for bias
            m_hat_bias = self.m_bias[i] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second moment estimate for bias
            v_hat_bias = self.v_bias[i] / (1 - self.beta2 ** self.t)

            # Update biases
            layer.bias -= self.learning_rate * m_hat_bias / (np.sqrt(v_hat_bias) + self.epsilon)

# Training and Evaluating the Manual MLP
def train_and_evaluate_manual_mlp(X_train, T_train, X_test, T_test, input_size, hidden_size, output_size, epochs, learning_rate, l2_reg=0.0, batch_size=32, optimiser = 'SGD'):
    """Train and evaluate a manual MLP model."""
    
    # Define the layers of the MLP
    layers = [
        Linear(input_size, hidden_size, l2_reg=l2_reg),
        ReLu(),
        Dropout(rate=0.5),
        Linear(hidden_size, output_size, l2_reg=l2_reg)
    ]

    # Initialize the feed-forward network with the defined layers
    mlp = FeedForwardNetwork(layers)
    # Initialize the cross-entropy loss
    ce_loss = CrossEntropy()
    # Initialize the SGD optimizer
    if optimiser == 'SGD':
        optimizer = SGDOptimizer(mlp.layers, learning_rate=learning_rate)
    elif optimiser == 'Adam':
        optimizer = AdamOptimizer(mlp.layers, learning_rate=learning_rate)

    # Lists to store training and test accuracy and loss
    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []

    for epoch in range(epochs):
        # Shuffle the training data at the beginning of each epoch
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        T_train = T_train[indices]

        for start_idx in range(0, X_train.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, X_train.shape[0])
            X_batch = X_train[start_idx:end_idx]
            T_batch = T_train[start_idx:end_idx]

            # Forward pass
            output = mlp.forward(X_batch, train=True)
            
            # Compute loss
            loss = ce_loss.forward(output, T_batch)
            
            # Backward pass (compute gradients)
            dldy = ce_loss.backward(output, T_batch)
            mlp.backward(dldy)

            # Update parameters with optimizer
            optimizer.update()

        # Evaluate on training and test sets
        train_output = mlp.forward(X_train, train=False)
        test_output = mlp.forward(X_test, train=False)
        train_acc = accuracy(train_output, T_train)
        test_acc = accuracy(test_output, T_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        train_loss_list.append(loss)
        test_loss_list.append(ce_loss.forward(test_output, T_test))

        # Add debugging information
        test_loss = ce_loss.forward(test_output, T_test)
        print(f'Epoch {epoch + 1}, Loss: {loss}, Train Accuracy: {train_acc}, Test Accuracy: {test_acc}')
        print(f'| Test NLL: {test_loss:.6f} | Test Acc: {test_acc * 100:.2f}%')

    return mlp, train_acc_list, test_acc_list, train_loss_list, test_loss_list

# Training and Evaluating the Scikit-learn MLP
def train_and_evaluate_sklearn_mlp(X_train, T_train, X_test, T_test, hidden_layer_sizes, max_iter, learning_rate_init, alpha):
    """Train and evaluate an MLP model using scikit-learn."""
    # Initialize the MLP classifier with the specified parameters
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, learning_rate_init=learning_rate_init, alpha=alpha, random_state=1)
    # Fit the MLP classifier to the training data
    mlp.fit(X_train, T_train)
    
    # Make predictions on the training and test sets
    train_predictions = mlp.predict(X_train)
    test_predictions = mlp.predict(X_test)
    
    # Compute accuracy for training and test sets
    train_accuracy = accuracy_score(T_train, train_predictions)
    test_accuracy = accuracy_score(T_test, test_predictions)
    
    return train_accuracy, test_accuracy, mlp  # Return the accuracies and the trained model
