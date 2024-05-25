import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logistic_regression import predict_logistic
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def plot_confusion_matrix(y_true, y_pred, title):
    """Plot and display the confusion matrix with a given title."""
    cm = confusion_matrix(y_true, y_pred)  # Compute the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)  # Create a display object for the confusion matrix
    disp.plot()  # Plot the confusion matrix
    plt.title(title)  # Set the title of the plot
    plt.show()  # Display the plot

def generate_confusion_matrices_mlp(mlp_manual, model_sklearn, X_train, T_train, X_test, T_test, dataset_name):
    """Generate and plot confusion matrices for both manual and scikit-learn MLP models."""
    # Generate predictions for the manual MLP model on the training set
    y_train_pred_manual = np.argmax(mlp_manual.forward(X_train, train=False), axis=1)
    # Generate predictions for the manual MLP model on the test set
    y_test_pred_manual = np.argmax(mlp_manual.forward(X_test, train=False), axis=1)
    # Plot confusion matrix for the manual MLP model on the training set
    plot_confusion_matrix(T_train, y_train_pred_manual, f"Confusion Matrix - Manual MLP - {dataset_name} Train")
    # Plot confusion matrix for the manual MLP model on the test set
    plot_confusion_matrix(T_test, y_test_pred_manual, f"Confusion Matrix - Manual MLP - {dataset_name} Test")
    
    # Generate predictions for the scikit-learn MLP model on the training set
    y_train_pred_sklearn = model_sklearn.predict(X_train)
    # Generate predictions for the scikit-learn MLP model on the test set
    y_test_pred_sklearn = model_sklearn.predict(X_test)
    # Plot confusion matrix for the scikit-learn MLP model on the training set
    plot_confusion_matrix(T_train, y_train_pred_sklearn, f"Confusion Matrix - Scikit-learn MLP - {dataset_name} Train")
    # Plot confusion matrix for the scikit-learn MLP model on the test set
    plot_confusion_matrix(T_test, y_test_pred_sklearn, f"Confusion Matrix - Scikit-learn MLP - {dataset_name} Test")
    
def generate_confusion_matrices_logreg(w_manual, model_sklearn, X_train, T_train, X_test, T_test, dataset_name):
    """Generate and plot confusion matrices for both manual and scikit-learn logistic regression models."""
    # Generate predictions for the manual logistic regression model on the training set
    y_train_pred_manual = (predict_logistic(X_train, w_manual) >= 0.5).astype(int)
    # Generate predictions for the manual logistic regression model on the test set
    y_test_pred_manual = (predict_logistic(X_test, w_manual) >= 0.5).astype(int)
    # Plot confusion matrix for the manual logistic regression model on the training set
    plot_confusion_matrix(T_train, y_train_pred_manual, f"Confusion Matrix - Manual LogReg - {dataset_name} Train")
    # Plot confusion matrix for the manual logistic regression model on the test set
    plot_confusion_matrix(T_test, y_test_pred_manual, f"Confusion Matrix - Manual LogReg - {dataset_name} Test")
    
    # Generate predictions for the scikit-learn logistic regression model on the training set
    y_train_pred_sklearn = model_sklearn.predict(X_train)
    # Generate predictions for the scikit-learn logistic regression model on the test set
    y_test_pred_sklearn = model_sklearn.predict(X_test)
    # Plot confusion matrix for the scikit-learn logistic regression model on the training set
    plot_confusion_matrix(T_train, y_train_pred_sklearn, f"Confusion Matrix - Scikit-learn LogReg - {dataset_name} Train")
    # Plot confusion matrix for the scikit-learn logistic regression model on the test set
    plot_confusion_matrix(T_test, y_test_pred_sklearn, f"Confusion Matrix - Scikit-learn LogReg - {dataset_name} Test")

def print_classification_report(y_true, y_pred, title):
    """Print and return the classification report as a DataFrame."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)  # Generate the classification report
    df = pd.DataFrame(report).transpose()  # Convert the report to a DataFrame
    print(f"{title}\n{df}\n")  # Print the title and the classification report
    return df  # Return the classification report DataFrame

def generate_classification_reports_mlp(mlp_manual, model_sklearn, X_train, T_train, X_test, T_test, dataset_name):
    """Generate and print classification reports for both manual and scikit-learn MLP models."""
    # Generate predictions for the manual MLP model on the training set
    y_train_pred_manual = np.argmax(mlp_manual.forward(X_train, train=False), axis=1)
    # Generate predictions for the manual MLP model on the test set
    y_test_pred_manual = np.argmax(mlp_manual.forward(X_test, train=False), axis=1)
    # Print and get the classification report for the manual MLP model on the training set
    report_train_manual = print_classification_report(T_train, y_train_pred_manual, f"Manual MLP - {dataset_name} Train")
    # Print and get the classification report for the manual MLP model on the test set
    report_test_manual = print_classification_report(T_test, y_test_pred_manual, f"Manual MLP - {dataset_name} Test")
    
    # Generate predictions for the scikit-learn MLP model on the training set
    y_train_pred_sklearn = model_sklearn.predict(X_train)
    # Generate predictions for the scikit-learn MLP model on the test set
    y_test_pred_sklearn = model_sklearn.predict(X_test)
    # Print and get the classification report for the scikit-learn MLP model on the training set
    report_train_sklearn = print_classification_report(T_train, y_train_pred_sklearn, f"Scikit-learn MLP - {dataset_name} Train")
    # Print and get the classification report for the scikit-learn MLP model on the test set
    report_test_sklearn = print_classification_report(T_test, y_test_pred_sklearn, f"Scikit-learn MLP - {dataset_name} Test")
    
    return report_train_manual, report_test_manual, report_train_sklearn, report_test_sklearn

def generate_classification_reports_logreg(w_manual, model_sklearn, X_train, T_train, X_test, T_test, dataset_name):
    """Generate and print classification reports for both manual and scikit-learn logistic regression models."""
    # Generate predictions for the manual logistic regression model on the training set
    y_train_pred_manual = (predict_logistic(X_train, w_manual) >= 0.5).astype(int)
    # Generate predictions for the manual logistic regression model on the test set
    y_test_pred_manual = (predict_logistic(X_test, w_manual) >= 0.5).astype(int)
    # Print and get the classification report for the manual logistic regression model on the training set
    report_train_manual = print_classification_report(T_train, y_train_pred_manual, f"Manual LogReg - {dataset_name} Train")
    # Print and get the classification report for the manual logistic regression model on the test set
    report_test_manual = print_classification_report(T_test, y_test_pred_manual, f"Manual LogReg - {dataset_name} Test")
    
    # Generate predictions for the scikit-learn logistic regression model on the training set
    y_train_pred_sklearn = model_sklearn.predict(X_train)
    # Generate predictions for the scikit-learn logistic regression model on the test set
    y_test_pred_sklearn = model_sklearn.predict(X_test)
    # Print and get the classification report for the scikit-learn logistic regression model on the training set
    report_train_sklearn = print_classification_report(T_train, y_train_pred_sklearn, f"Scikit-learn LogReg - {dataset_name} Train")
    # Print and get the classification report for the scikit-learn logistic regression model on the test set
    report_test_sklearn = print_classification_report(T_test, y_test_pred_sklearn, f"Scikit-learn LogReg - {dataset_name} Test")
    
    return report_train_manual, report_test_manual, report_train_sklearn, report_test_sklearn

def plot_learning_curves(train_acc, test_acc, train_loss, test_loss, title):
    """Plot learning curves for training and test accuracy and loss."""
    epochs = range(1, len(train_acc) + 1)  # Generate a range of epochs
    plt.figure(figsize=(14, 6))  # Create a figure for the plots

    plt.subplot(1, 2, 1)  # Create a subplot for accuracy
    plt.plot(epochs, train_acc, 'bo-', label='Train Accuracy')  # Plot training accuracy
    plt.plot(epochs, test_acc, 'ro-', label='Test Accuracy')  # Plot test accuracy
    plt.title(f'{title} - Accuracy')  # Set the title for the accuracy plot
    plt.xlabel('Epochs')  # Label the x-axis
    plt.ylabel('Accuracy')  # Label the y-axis
    plt.legend()  # Add a legend

    plt.subplot(1, 2, 2)  # Create a subplot for loss
    plt.plot(epochs, train_loss, 'bo-', label='Train Loss')  # Plot training loss
    plt.plot(epochs, test_loss, 'ro-', label='Test Loss')  # Plot test loss
    plt.title(f'{title} - Loss')  # Set the title for the loss plot
    plt.xlabel('Epochs')  # Label the x-axis
    plt.ylabel('Loss')  # Label the y-axis
    plt.legend()  # Add a legend

    plt.tight_layout()  # Adjust subplots to fit into the figure area
    plt.show()  # Display the plots

def plot_all_learning_curves(train_acc_avc_manual, test_acc_avc_manual, train_loss_avc_manual, test_loss_avc_manual, 
                                train_acc_salary_manual, test_acc_salary_manual, train_loss_salary_manual, test_loss_salary_manual, 
                                train_acc_avc_sklearn, test_acc_avc_sklearn, 
                                train_acc_salary_sklearn, test_acc_salary_sklearn, algorithm_name):
    """Plot learning curves for both manual and scikit-learn models for AVC and Salary datasets."""
    # Plot learning curves for the manual model on the AVC dataset
    plot_name = f"Manual {algorithm_name} - AVC"
    plot_learning_curves(train_acc_avc_manual, test_acc_avc_manual, train_loss_avc_manual, test_loss_avc_manual, plot_name)
    # Plot learning curves for the manual model on the Salary dataset
    plot_name = f"Manual {algorithm_name} - Salary"
    plot_learning_curves(train_acc_salary_manual, test_acc_salary_manual, train_loss_salary_manual, test_loss_salary_manual, plot_name)

def generate_comparative_table(reports, dataset_name, algorithm_name):
    """Generate and print a comparative table of classification metrics for different algorithms."""
    metrics = ['precision', 'recall', 'f1-score']  # Metrics to include in the table
    table = pd.DataFrame(columns=['Algorithm', 'Dataset'] + metrics)  # Initialize an empty DataFrame
    
    for algo, (train_report, test_report) in reports.items():
        # Extract macro average metrics for train and test datasets
        train_row = [f'{algo} Train', algorithm_name]
        test_row = [f'{algo} Test', algorithm_name]
        for metric in metrics:
            train_row.append(train_report.loc['macro avg', metric])  # Append macro average train metric
            test_row.append(test_report.loc['macro avg', metric])  # Append macro average test metric
        table = pd.concat([table, pd.Series(train_row, index=table.columns).to_frame().T], ignore_index=True)  # Add train row to table
        table = pd.concat([table, pd.Series(test_row, index=table.columns).to_frame().T], ignore_index=True)  # Add test row to table
    
    # Highlight maximum values for each metric
    for metric in metrics:
        max_idx = table[metric].idxmax()  # Get the index of the maximum value
        table.loc[max_idx, metric] = f"**{table.loc[max_idx, metric]:.4f}**"  # Highlight the maximum value
    
    print(f"Comparative Table for {dataset_name}\n{table}\n")  # Print the comparative table
    table.to_csv(f'output/{dataset_name}_{algorithm_name}_comparative_table.csv', index=False)  # Save the table to a CSV file

