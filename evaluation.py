import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logistic_regression import predict_logistic
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(title)
    plt.show()

def generate_confusion_matrices_mlp(mlp_manual, model_sklearn, X_train, T_train, X_test, T_test, dataset_name):
    y_train_pred_manual = np.argmax(mlp_manual.forward(X_train, train=False), axis=1)
    y_test_pred_manual = np.argmax(mlp_manual.forward(X_test, train=False), axis=1)
    plot_confusion_matrix(T_train, y_train_pred_manual, f"Confusion Matrix - Manual MLP - {dataset_name} Train")
    plot_confusion_matrix(T_test, y_test_pred_manual, f"Confusion Matrix - Manual MLP - {dataset_name} Test")
    
    y_train_pred_sklearn = model_sklearn.predict(X_train)
    y_test_pred_sklearn = model_sklearn.predict(X_test)
    plot_confusion_matrix(T_train, y_train_pred_sklearn, f"Confusion Matrix - Scikit-learn MLP - {dataset_name} Train")
    plot_confusion_matrix(T_test, y_test_pred_sklearn, f"Confusion Matrix - Scikit-learn MLP - {dataset_name} Test")
    
def generate_confusion_matrices_logreg(w_manual, model_sklearn, X_train, T_train, X_test, T_test, dataset_name):
    # Generate predictions for manual logistic regression model
    y_train_pred_manual = (predict_logistic(X_train, w_manual) >= 0.5).astype(int)
    y_test_pred_manual = (predict_logistic(X_test, w_manual) >= 0.5).astype(int)
    
    plot_confusion_matrix(T_train, y_train_pred_manual, f"Confusion Matrix - Manual LogReg - {dataset_name} Train")
    plot_confusion_matrix(T_test, y_test_pred_manual, f"Confusion Matrix - Manual LogReg - {dataset_name} Test")
    
    # Generate predictions for scikit-learn logistic regression model
    y_train_pred_sklearn = model_sklearn.predict(X_train)
    y_test_pred_sklearn = model_sklearn.predict(X_test)
    
    plot_confusion_matrix(T_train, y_train_pred_sklearn, f"Confusion Matrix - Scikit-learn LogReg - {dataset_name} Train")
    plot_confusion_matrix(T_test, y_test_pred_sklearn, f"Confusion Matrix - Scikit-learn LogReg - {dataset_name} Test")


def print_classification_report(y_true, y_pred, title):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose()
    print(f"{title}\n{df}\n")
    return df

def generate_classification_reports_mlp(mlp_manual, model_sklearn, X_train, T_train, X_test, T_test, dataset_name):
    y_train_pred_manual = np.argmax(mlp_manual.forward(X_train, train=False), axis=1)
    y_test_pred_manual = np.argmax(mlp_manual.forward(X_test, train=False), axis=1)
    report_train_manual = print_classification_report(T_train, y_train_pred_manual, f"Manual MLP - {dataset_name} Train")
    report_test_manual = print_classification_report(T_test, y_test_pred_manual, f"Manual MLP - {dataset_name} Test")
    
    y_train_pred_sklearn = model_sklearn.predict(X_train)
    y_test_pred_sklearn = model_sklearn.predict(X_test)
    report_train_sklearn = print_classification_report(T_train, y_train_pred_sklearn, f"Scikit-learn MLP - {dataset_name} Train")
    report_test_sklearn = print_classification_report(T_test, y_test_pred_sklearn, f"Scikit-learn MLP - {dataset_name} Test")
    
    return report_train_manual, report_test_manual, report_train_sklearn, report_test_sklearn

def generate_classification_reports_logreg(w_manual, model_sklearn, X_train, T_train, X_test, T_test, dataset_name):
    y_train_pred_manual = (predict_logistic(X_train, w_manual) >= 0.5).astype(int)
    y_test_pred_manual = (predict_logistic(X_test, w_manual) >= 0.5).astype(int)
    report_train_manual = print_classification_report(T_train, y_train_pred_manual, f"Manual LogReg - {dataset_name} Train")
    report_test_manual = print_classification_report(T_test, y_test_pred_manual, f"Manual LogReg - {dataset_name} Test")
    
    y_train_pred_sklearn = model_sklearn.predict(X_train)
    y_test_pred_sklearn = model_sklearn.predict(X_test)
    report_train_sklearn = print_classification_report(T_train, y_train_pred_sklearn, f"Scikit-learn LogReg - {dataset_name} Train")
    report_test_sklearn = print_classification_report(T_test, y_test_pred_sklearn, f"Scikit-learn LogReg - {dataset_name} Test")
    
    return report_train_manual, report_test_manual, report_train_sklearn, report_test_sklearn


def plot_learning_curves(train_acc, test_acc, train_loss, test_loss, title):
    epochs = range(1, len(train_acc) + 1)
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, 'bo-', label='Train Accuracy')
    plt.plot(epochs, test_acc, 'ro-', label='Test Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, 'bo-', label='Train Loss')
    plt.plot(epochs, test_loss, 'ro-', label='Test Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
def plot_all_learning_curves(train_acc_avc_manual, test_acc_avc_manual, train_loss_avc_manual, test_loss_avc_manual, 
                                train_acc_salary_manual, test_acc_salary_manual, train_loss_salary_manual, test_loss_salary_manual, 
                                train_acc_avc_sklearn, test_acc_avc_sklearn, 
                                train_acc_salary_sklearn, test_acc_salary_sklearn, algorithm_name):
    plot_name = f"Manual {algorithm_name} - AVC"
    plot_learning_curves(train_acc_avc_manual, test_acc_avc_manual, train_loss_avc_manual, test_loss_avc_manual, plot_name)
    plot_name = f"Manual {algorithm_name} - Salary"
    plot_learning_curves(train_acc_salary_manual, test_acc_salary_manual, train_loss_salary_manual, test_loss_salary_manual, plot_name)

def generate_comparative_table(reports, dataset_name, algorithm_name):
    metrics = ['precision', 'recall', 'f1-score']
    table = pd.DataFrame(columns=['Algorithm', 'Class'] + metrics)
    
    for algo, (train_report, test_report) in reports.items():
        for class_label in train_report.index[:-3]:  # Exclude 'accuracy', 'macro avg', 'weighted avg'
            train_row = [f'{algo} Train', class_label]
            test_row = [f'{algo} Test', class_label]
            for metric in metrics:
                train_row.append(train_report.loc[class_label, metric])
                test_row.append(test_report.loc[class_label, metric])
            table = pd.concat([table, pd.Series(train_row, index=table.columns).to_frame().T], ignore_index=True)
            table = pd.concat([table, pd.Series(test_row, index=table.columns).to_frame().T], ignore_index=True)
    
    # Highlight maximum values for each metric
    for metric in metrics:
        max_idx = table[metric].idxmax()
        table.loc[max_idx, metric] = f"**{table.loc[max_idx, metric]:.4f}**"
    
    print(f"Comparative Table for {dataset_name}\n{table}\n")
    table.to_csv(f'output/{dataset_name}_{algorithm_name}_comparative_table.csv', index=False)