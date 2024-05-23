import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from imblearn.over_sampling import SMOTE

def logistic(x):
    x = np.array(x, dtype=float)  # Ensure x is a numpy array with float type
    return 1 / (1 + np.exp(-x))

def nll(Y, T):
    N = T.shape[0]
    return -np.sum(T * np.log(Y) + (1 - T) * np.log(1 - Y)) / N

def accuracy(Y, T):
    N = Y.shape[0]
    acc = 0
    for i in range(N):
        if (Y[i] >= 0.5 and T[i] == 1) or (Y[i] < 0.5 and T[i] == 0):
            acc += 1
    return acc / N

def predict_logistic(X, w):
    return logistic(np.dot(X, w))

def train_and_eval_logistic(X_train, T_train, X_test, T_test, lr=0.01, epochs_no=100):
    (N, D) = X_train.shape
    
    # Initial weights
    w = np.random.randn(D)
    
    train_acc, test_acc = [], []
    train_nll, test_nll = [], []

    for epoch in range(epochs_no):
        Y_train = predict_logistic(X_train, w)
        Y_test = predict_logistic(X_test, w)

        train_acc.append(accuracy(Y_train, T_train))
        test_acc.append(accuracy(Y_test, T_test))
        train_nll.append(nll(Y_train, T_train))
        test_nll.append(nll(Y_test, T_test))

        w = w - lr * np.dot(X_train.T, Y_train - T_train) / N
        
    return w, train_nll, test_nll, train_acc, test_acc


def train_and_eval_sklearn_logistic(X_train, T_train, X_test, T_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, T_train)
    
    Y_train = model.predict_proba(X_train)[:, 1]
    Y_test = model.predict_proba(X_test)[:, 1]
    
    train_acc = accuracy_score(T_train, (Y_train >= 0.5).astype(int))
    test_acc = accuracy_score(T_test, (Y_test >= 0.5).astype(int))
    train_nll = log_loss(T_train, Y_train)
    test_nll = log_loss(T_test, Y_test)
    
    return model, train_nll, test_nll, train_acc, test_acc
