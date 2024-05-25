import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

def logistic(x):
    """Apply the logistic function to the input array x."""
    x = np.array(x, dtype=float)  # Ensure x is a numpy array with float type
    return 1 / (1 + np.exp(-x))  # Compute the logistic function

def nll(Y, T):
    """Compute the negative log-likelihood (NLL) for binary classification."""
    N = T.shape[0]  # Number of samples
    return -np.sum(T * np.log(Y) + (1 - T) * np.log(1 - Y)) / N  # Calculate NLL

def accuracy(Y, T):
    """Compute the accuracy of predictions Y against true labels T."""
    N = Y.shape[0]  # Number of samples
    acc = 0  # Initialize accuracy counter
    for i in range(N):
        # Increment accuracy counter if prediction matches the true label
        if (Y[i] >= 0.5 and T[i] == 1) or (Y[i] < 0.5 and T[i] == 0):
            acc += 1
    return acc / N  # Return accuracy

def predict_logistic(X, w):
    """Make predictions using logistic regression with weights w."""
    return logistic(np.dot(X, w))  # Apply logistic function to linear combination of inputs

def train_and_eval_logistic(X_train, T_train, X_test, T_test, lr=0.01, epochs_no=100):
    """Train and evaluate a logistic regression model."""
    (N, D) = X_train.shape  # Get number of samples (N) and number of features (D)
    
    # Initialize weights randomly
    w = np.random.randn(D)
    
    train_acc, test_acc = [], []  # Lists to store training and test accuracy
    train_nll, test_nll = [], []  # Lists to store training and test NLL

    for epoch in range(epochs_no):
        # Make predictions on the training and test sets
        Y_train = predict_logistic(X_train, w)
        Y_test = predict_logistic(X_test, w)

        # Append current accuracy and NLL to the lists
        train_acc.append(accuracy(Y_train, T_train))
        test_acc.append(accuracy(Y_test, T_test))
        train_nll.append(nll(Y_train, T_train))
        test_nll.append(nll(Y_test, T_test))

        # Update weights using gradient descent
        w = w - lr * np.dot(X_train.T, Y_train - T_train) / N
        
    return w, train_nll, test_nll, train_acc, test_acc  # Return final weights and performance metrics

def train_and_eval_sklearn_logistic(X_train, T_train, X_test, T_test, penalty='l2', C=1.0, solver='liblinear'):
    """Train and evaluate a logistic regression model using scikit-learn."""
    model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=1000)  # Initialize logistic regression model
    model.fit(X_train, T_train)  # Fit the model to the training data
    
    # Get predicted probabilities for the training and test sets
    Y_train = model.predict_proba(X_train)[:, 1]
    Y_test = model.predict_proba(X_test)[:, 1]
    
    # Compute accuracy for training and test sets
    train_acc = accuracy_score(T_train, (Y_train >= 0.5).astype(int))
    test_acc = accuracy_score(T_test, (Y_test >= 0.5).astype(int))
    # Compute NLL for training and test sets
    train_nll = log_loss(T_train, Y_train)
    test_nll = log_loss(T_test, Y_test)
    
    return model, train_nll, test_nll, train_acc, test_acc  # Return the model and performance metrics

