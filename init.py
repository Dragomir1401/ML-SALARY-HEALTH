import os
import pandas as pd
import numpy as np
from mlp import train_and_evaluate_manual_mlp, train_and_evaluate_sklearn_mlp
from logistic_regression import train_and_eval_logistic, train_and_eval_sklearn_logistic
from evaluation import generate_confusion_matrices_mlp, generate_confusion_matrices_logreg, generate_classification_reports_logreg, generate_classification_reports_mlp, plot_all_learning_curves, generate_comparative_table
from preprocess import impute_missing_values, handle_extreme_values, remove_redundant_attributes, preprocess_data, standardize_data, encode_categorical
from visualisation import analyze_attributes, numeric_statistics, plot_boxplot, categorical_statistics, plot_histograms, plot_class_balance, plot_correlation_matrix, plot_categorical_correlation_matrix

# Reading the CSV files
avc_df_full = pd.read_csv('tema2_AVC/AVC_full.csv')
avc_df_train = pd.read_csv('tema2_AVC/AVC_train.csv')
avc_df_test = pd.read_csv('tema2_AVC/AVC_test.csv')

# Reading the CSV files
salary_df_full = pd.read_csv('tema2_SalaryPrediction/SalaryPrediction_full.csv')
salary_df_train = pd.read_csv('tema2_SalaryPrediction/SalaryPrediction_train.csv')
salary_df_test = pd.read_csv('tema2_SalaryPrediction/SalaryPrediction_test.csv')

# Numeric statistics
numeric_columns_avc = ['mean_blood_sugar_level', 'body_mass_indicator', 'years_old', 'analysis_results', 'biological_age_index']
numeric_columns_salary = ['fnl', 'hpw', 'gain', 'edu_int', 'years', 'loss', 'prod']

# Categorical statistics
categorical_columns_avc = ['cardiovascular_issues', 'job_category', 'sex', 'tobacco_usage', 'high_blood_pressure', 'married', 'living_area', 'chaotic_sleep', 'cerebrovascular_accident']
categorical_columns_salary = ['relation', 'country', 'job', 'work_type', 'partner', 'edu', 'gender', 'race', 'gtype', 'money']

# Define the class columns for each dataset
class_column_avc = 'cerebrovascular_accident'  # This is the class column in the AVC dataset
class_column_salary = 'money'  # This is the class column in the Salary dataset

# Preprocessing for full, train, and test datasets
datasets = {
    'avc_full': avc_df_full,
    'avc_train': avc_df_train,
    'avc_test': avc_df_test,
    'salary_full': salary_df_full,
    'salary_train': salary_df_train,
    'salary_test': salary_df_test
}

def analyze_wrapper():
    # Analyze attributes
    avc_attribute_info = analyze_attributes(avc_df_full)
    salary_attribute_info = analyze_attributes(salary_df_full)
    
    # Open 2 files in dir output/
    with open('output/avc_attribute_info.txt', 'w') as f:
        f.write(avc_attribute_info.to_string())
        
    with open('output/salary_attribute_info.txt', 'w') as f:
        f.write(salary_attribute_info.to_string())
        
def numeric_statistics_wrapper():
    avc_numeric_statistics = numeric_statistics(avc_df_full, numeric_columns_avc)
    salary_numeric_statistics = numeric_statistics(salary_df_full, numeric_columns_salary)

    # Write numeric statistics to files
    with open('output/avc_numeric_statistics.txt', 'w') as f:
        f.write(avc_numeric_statistics.to_string())
        
    with open('output/salary_numeric_statistics.txt', 'w') as f:
        f.write(salary_numeric_statistics.to_string())
        
def boxplots_wrapper():
    # Boxplot for numeric attributes
    plot_boxplot(avc_df_full, numeric_columns_avc, 'AVC - Boxplot for Numeric Attributes')
    plot_boxplot(salary_df_full, numeric_columns_salary, 'Salary - Boxplot for Numeric Attributes')
    
def categorial_statistics_wrapper():
    avc_categorical_statistics = categorical_statistics(avc_df_full, categorical_columns_avc)
    salary_categorical_statistics = categorical_statistics(salary_df_full, categorical_columns_salary)
    
    # Display categorical statistics into files
    with open('output/avc_categorical_statistics.txt', 'w') as f:
        f.write(avc_categorical_statistics.to_string())
        
    with open('output/salary_categorical_statistics.txt', 'w') as f:
        f.write(salary_categorical_statistics.to_string())
        
def histograms_wrapper():
    # Histograms for categorical attributes and save them
    plot_histograms(avc_df_full, categorical_columns_avc, 'AVC - Histograms for Categorical Attributes')
    plot_histograms(salary_df_full, categorical_columns_salary, 'Salary - Histograms for Categorical Attributes')
    
def class_balance_wrapper():
    # Plot class balance using seaborn for each dataset
    plot_class_balance(avc_df_full, class_column_avc, 'AVC Dataset Full')
    plot_class_balance(avc_df_train, class_column_avc, 'AVC Dataset Train')
    plot_class_balance(avc_df_train, class_column_avc, 'AVC Dataset Test')
    plot_class_balance(salary_df_full, class_column_salary, 'Salary Dataset Full')
    plot_class_balance(salary_df_train, class_column_salary, 'Salary Dataset Train')
    plot_class_balance(salary_df_test, class_column_salary, 'Salary Dataset Test')
    
def correlation_matrix_wrapper():
    # Plot correlation matrices for numerical attributes
    plot_correlation_matrix(avc_df_full, numeric_columns_avc, 'AVC Dataset Full')
    plot_correlation_matrix(salary_df_full, numeric_columns_salary, 'Salary Dataset Full')
    plot_correlation_matrix(avc_df_train, numeric_columns_avc, 'AVC Dataset Train')
    plot_correlation_matrix(salary_df_train, numeric_columns_salary, 'Salary Dataset Train')
    plot_correlation_matrix(avc_df_test, numeric_columns_avc, 'AVC Dataset Test')
    plot_correlation_matrix(salary_df_test, numeric_columns_salary, 'Salary Dataset Test')
    
def categorial_correlation_matrix_wrapper():
    # Plot correlation matrices for categorical attributes
    plot_categorical_correlation_matrix(avc_df_full, categorical_columns_avc, 'AVC Dataset Full')
    plot_categorical_correlation_matrix(salary_df_full, categorical_columns_salary, 'Salary Dataset Full')
    plot_categorical_correlation_matrix(avc_df_train, categorical_columns_avc, 'AVC Dataset Train')
    plot_categorical_correlation_matrix(salary_df_train, categorical_columns_salary, 'Salary Dataset Train')
    plot_categorical_correlation_matrix(avc_df_test, categorical_columns_avc, 'AVC Dataset Test')
    plot_categorical_correlation_matrix(salary_df_test, categorical_columns_salary, 'Salary Dataset Test')
    
def preprocess_data_wrapper():
    processed_datasets = {}
    
    for name, df in datasets.items():
        print(f"Processing dataset: {name}")
        
        # Check if target column is present in the dataset
        target_column = class_column_avc if 'avc' in name else class_column_salary
        if target_column not in df.columns:
            raise KeyError(f"Target column '{target_column}' not found in dataset '{name}'")

        # Separate columns
        numeric_columns = numeric_columns_avc if 'avc' in name else numeric_columns_salary
        categorical_columns = categorical_columns_avc if 'avc' in name else categorical_columns_salary
        
        # First eliminate highly correlated attributes
        df_reduced, dropped_columns = remove_redundant_attributes(df, numeric_columns, threshold=0.5)
        print(f"Dropped columns in {name}: {dropped_columns}")

        # Make a copy of numeric columns without the dropped columns
        numeric_columns_after_drop = [col for col in numeric_columns if col not in dropped_columns]

        # Impute missing values
        df_imputed = impute_missing_values(df_reduced, numeric_columns_after_drop, categorical_columns)
        if target_column not in df_imputed.columns:
            raise KeyError(f"Target column '{target_column}' missing after imputation in dataset '{name}'")

        # Handle extreme values
        df_outliers_handled = handle_extreme_values(df_imputed, numeric_columns_after_drop)
        if target_column not in df_outliers_handled.columns:
            raise KeyError(f"Target column '{target_column}' missing after handling extreme values in dataset '{name}'")

        # Impute missing values again after handling extreme values
        df_outliers_imputed = impute_missing_values(df_outliers_handled, numeric_columns_after_drop, categorical_columns)
        if target_column not in df_outliers_imputed.columns:
            raise KeyError(f"Target column '{target_column}' missing after second imputation in dataset '{name}'")

        # Standardize numerical attributes
        df_standardized = standardize_data(df_outliers_imputed, numeric_columns_after_drop, method='standard')
        if target_column not in df_standardized.columns:
            raise KeyError(f"Target column '{target_column}' missing after standardization in dataset '{name}'")

        # Save the processed dataframe for logistic regression
        processed_datasets[name] = df_standardized
        
        # Write the processed data to files for testing
        df_standardized.to_csv(f'output/{name}_processed.csv', index=False)
    
    # Encode categorical variables and prepare datasets for logistic regression
    X_avc_full, T_avc_full, label_encoder_avc, onehot_encoder_avc = preprocess_data(processed_datasets['avc_full'], class_column_avc)
    X_avc_train, T_avc_train, _, _ = preprocess_data(processed_datasets['avc_train'], class_column_avc, encoder=onehot_encoder_avc)
    X_avc_test, T_avc_test, _, _ = preprocess_data(processed_datasets['avc_test'], class_column_avc, encoder=onehot_encoder_avc)

    X_salary_full, T_salary_full, label_encoder_salary, onehot_encoder_salary = preprocess_data(processed_datasets['salary_full'], class_column_salary)
    X_salary_train, T_salary_train, _, _ = preprocess_data(processed_datasets['salary_train'], class_column_salary, encoder=onehot_encoder_salary)
    X_salary_test, T_salary_test, _, _ = preprocess_data(processed_datasets['salary_test'], class_column_salary, encoder=onehot_encoder_salary)
    
    return_tuple_avc = (X_avc_full, T_avc_full, label_encoder_avc, onehot_encoder_avc, X_avc_train, T_avc_train, X_avc_test, T_avc_test)
    return_tuple_salary = (X_salary_full, T_salary_full, label_encoder_salary, onehot_encoder_salary, X_salary_train, T_salary_train, X_salary_test, T_salary_test)

    return return_tuple_avc, return_tuple_salary


def logistic_regression_wrapper(X_avc_train, T_avc_train, X_avc_test, T_avc_test, X_salary_train, T_salary_train, X_salary_test, T_salary_test):
    # Manual Logistic Regression
    w_avc, train_nll_avc, test_nll_avc, train_acc_avc, test_acc_avc = train_and_eval_logistic(
        X_avc_train, T_avc_train, X_avc_test, T_avc_test, lr=0.1, epochs_no=500)

    w_salary, train_nll_salary, test_nll_salary, train_acc_salary, test_acc_salary = train_and_eval_logistic(
        X_salary_train, T_salary_train, X_salary_test, T_salary_test, lr=0.1, epochs_no=500)

    # Logistic Regression using scikit-learn
    model_avc, train_nll_avc_sklearn, test_nll_avc_sklearn, train_acc_avc_sklearn, test_acc_avc_sklearn = train_and_eval_sklearn_logistic(
        X_avc_train, T_avc_train, X_avc_test, T_avc_test)

    model_salary, train_nll_salary_sklearn, test_nll_salary_sklearn, train_acc_salary_sklearn, test_acc_salary_sklearn = train_and_eval_sklearn_logistic(
        X_salary_train, T_salary_train, X_salary_test, T_salary_test)

    # Save logistic regression results
    with open('output/logistic_regression_results.txt', 'w') as f:
        f.write("Manual Logistic Regression Results:\n")
        f.write(f"AVC Dataset - Train Accuracy: {train_acc_avc[-1]}, Test Accuracy: {test_acc_avc[-1]}\n")
        f.write(f"Salary Dataset - Train Accuracy: {train_acc_salary[-1]}, Test Accuracy: {test_acc_salary[-1]}\n\n")

        f.write("Scikit-learn Logistic Regression Results:\n")
        f.write(f"AVC Dataset - Train Accuracy: {train_acc_avc_sklearn}, Test Accuracy: {test_acc_avc_sklearn}\n")
        f.write(f"Salary Dataset - Train Accuracy: {train_acc_salary_sklearn}, Test Accuracy: {test_acc_salary_sklearn}\n")
        
    return_tuple_avc = (w_avc, model_avc, train_acc_avc, test_acc_avc, train_nll_avc, test_nll_avc)
    return_tuple_salary = (w_salary, model_salary, train_acc_salary, test_acc_salary, train_nll_salary, test_nll_salary)
    return_tuple_sklearn = (train_acc_avc_sklearn, test_acc_avc_sklearn, train_acc_salary_sklearn, test_acc_salary_sklearn)
    
    return return_tuple_avc, return_tuple_salary, return_tuple_sklearn
        
def mlp_wrapper(X_avc_train, T_avc_train, X_avc_test, T_avc_test, X_salary_train, T_salary_train, X_salary_test, T_salary_test):
    # Define the MLP architecture and training parameters
    input_size_avc = X_avc_train.shape[1]
    input_size_salary = X_salary_train.shape[1]
    output_size = 2  # Binary classification

    hidden_size_avc = 128
    hidden_size_salary = 128
    epochs = 1000
    learning_rate = 0.00002
    l2_reg = 0.001
    batch_size = 16

    # Manual MLP Training and Evaluation for AVC dataset
    mlp_manual_avc, train_acc_avc_manual, test_acc_avc_manual, train_loss_avc_manual, test_loss_avc_manual = train_and_evaluate_manual_mlp(
        X_avc_train, T_avc_train, X_avc_test, T_avc_test, input_size_avc, hidden_size_avc, output_size, epochs, learning_rate, l2_reg, batch_size
    )
    
    # Manual MLP Training and Evaluation for Salary dataset
    mlp_manual_salary, train_acc_salary_manual, test_acc_salary_manual, train_loss_salary_manual, test_loss_salary_manual = train_and_evaluate_manual_mlp(
        X_salary_train, T_salary_train, X_salary_test, T_salary_test, input_size_salary, hidden_size_salary, output_size, epochs, learning_rate, l2_reg, batch_size
    )

    # Scikit-learn MLP Training and Evaluation for AVC dataset
    hidden_layer_sizes = (hidden_size_avc,)  # Single hidden layer example
    max_iter = 500
    learning_rate_init = 0.01
    alpha = 0.0001  # L2 regularization term

    train_acc_avc_sklearn, test_acc_avc_sklearn, model_avc_sklearn = train_and_evaluate_sklearn_mlp(
        X_avc_train, T_avc_train, X_avc_test, T_avc_test, hidden_layer_sizes, max_iter, learning_rate_init, alpha
    )

    # Scikit-learn MLP Training and Evaluation for Salary dataset
    hidden_layer_sizes = (hidden_size_salary,)  # Single hidden layer example

    train_acc_salary_sklearn, test_acc_salary_sklearn, model_salary_sklearn = train_and_evaluate_sklearn_mlp(
        X_salary_train, T_salary_train, X_salary_test, T_salary_test, hidden_layer_sizes, max_iter, learning_rate_init, alpha
    )

    # Save results
    with open('output/mlp_results.txt', 'w') as f:
        f.write("Manual MLP Results:\n")
        f.write(f"AVC Dataset - Train Accuracy: {train_acc_avc_manual[-1]}, Test Accuracy: {test_acc_avc_manual[-1]}\n")
        f.write(f"Salary Dataset - Train Accuracy: {train_acc_salary_manual[-1]}, Test Accuracy: {test_acc_salary_manual[-1]}\n\n")

        f.write("Scikit-learn MLP Results:\n")
        f.write(f"AVC Dataset - Train Accuracy: {train_acc_avc_sklearn}, Test Accuracy: {test_acc_avc_sklearn}\n")
        f.write(f"Salary Dataset - Train Accuracy: {train_acc_salary_sklearn}, Test Accuracy: {test_acc_salary_sklearn}\n")
        
    return_tuple_avc = (mlp_manual_avc, model_avc_sklearn, train_acc_avc_manual, test_acc_avc_manual, train_loss_avc_manual, test_loss_avc_manual)
    return_tuple_salary = (mlp_manual_salary, model_salary_sklearn, train_acc_salary_manual, test_acc_salary_manual, train_loss_salary_manual, test_loss_salary_manual)
    return_tuple_sklearn = (train_acc_avc_sklearn, test_acc_avc_sklearn, train_acc_salary_sklearn, test_acc_salary_sklearn)
    
    return return_tuple_avc, return_tuple_salary, return_tuple_sklearn


def process_and_generate_reports(return_tuple_avc, return_tuple_salary, return_tuple_sklearn, 
                                X_avc_train, T_avc_train, X_avc_test, T_avc_test, 
                                X_salary_train, T_salary_train, X_salary_test, T_salary_test, 
                                algorithm_name):
    # Parse return tuples
    model_manual_avc, model_sklearn_avc, train_acc_avc, test_acc_avc, train_nll_avc, test_nll_avc = return_tuple_avc
    model_manual_salary, model_sklearn_salary, train_acc_salary, test_acc_salary, train_nll_salary, test_nll_salary = return_tuple_salary
    train_acc_avc_sklearn, test_acc_avc_sklearn, train_acc_salary_sklearn, test_acc_salary_sklearn = return_tuple_sklearn
    
    # Generate and plot confusion matrices
    if algorithm_name == "MLP":
        generate_confusion_matrices_mlp(model_manual_avc, model_sklearn_avc, X_avc_train, T_avc_train, X_avc_test, T_avc_test, "AVC")
        generate_confusion_matrices_mlp(model_manual_salary, model_sklearn_salary, X_salary_train, T_salary_train, X_salary_test, T_salary_test, "Salary")
        
        report_train_manual_avc, report_test_manual_avc, report_train_sklearn_avc, report_test_sklearn_avc = generate_classification_reports_mlp(
            model_manual_avc, model_sklearn_avc, X_avc_train, T_avc_train, X_avc_test, T_avc_test, f"AVC {algorithm_name}")
        report_train_manual_salary, report_test_manual_salary, report_train_sklearn_salary, report_test_sklearn_salary = generate_classification_reports_mlp(
            model_manual_salary, model_sklearn_salary, X_salary_train, T_salary_train, X_salary_test, T_salary_test, f"Salary {algorithm_name}")
    elif algorithm_name == "LogReg":
        generate_confusion_matrices_logreg(model_manual_avc, model_sklearn_avc, X_avc_train, T_avc_train, X_avc_test, T_avc_test, "AVC")
        generate_confusion_matrices_logreg(model_manual_salary, model_sklearn_salary, X_salary_train, T_salary_train, X_salary_test, T_salary_test, "Salary")
        
        report_train_manual_avc, report_test_manual_avc, report_train_sklearn_avc, report_test_sklearn_avc = generate_classification_reports_logreg(
            model_manual_avc, model_sklearn_avc, X_avc_train, T_avc_train, X_avc_test, T_avc_test, f"AVC {algorithm_name}")
        report_train_manual_salary, report_test_manual_salary, report_train_sklearn_salary, report_test_sklearn_salary = generate_classification_reports_logreg(
            model_manual_salary, model_sklearn_salary, X_salary_train, T_salary_train, X_salary_test, T_salary_test, f"Salary {algorithm_name}")
    else:
        raise ValueError("Unsupported algorithm name")
    
    # Plot learning curves for avc and salary datasets
    plot_all_learning_curves(train_acc_avc, test_acc_avc, train_nll_avc, test_nll_avc,
                            train_acc_salary, test_acc_salary, train_nll_salary, test_nll_salary,
                            train_acc_avc_sklearn, test_acc_avc_sklearn,
                            train_acc_salary_sklearn, test_acc_salary_sklearn, algorithm_name)
    
    # Create dictionaries for classification reports
    reports_avc = {
        f'Manual {algorithm_name}': (report_train_manual_avc, report_test_manual_avc),
        f'Scikit-learn {algorithm_name}': (report_train_sklearn_avc, report_test_sklearn_avc),
    }
    
    reports_salary = {
        f'Manual {algorithm_name}': (report_train_manual_salary, report_test_manual_salary),
        f'Scikit-learn {algorithm_name}': (report_train_sklearn_salary, report_test_sklearn_salary),
    }
    
    # Generate comparative tables
    generate_comparative_table(reports_avc, 'AVC', algorithm_name)
    generate_comparative_table(reports_salary, 'Salary', algorithm_name)

def __main__():
    # Create output directory
    if not os.path.exists('output'):
        os.makedirs('output')
        
    # # Analyze attributes
    # analyze_wrapper()
    
    # # Numeric statistics
    # numeric_statistics_wrapper()

    # # Boxplots for numeric attributes
    # boxplots_wrapper()

    # # Categorial statistics
    # categorial_statistics_wrapper()

    # # Histograms for categorical attributes
    # histograms_wrapper()
    
    # # Class balance
    # class_balance_wrapper()
    
    # Correlation matrices for numerical attributes
    # correlation_matrix_wrapper()
    
    # # # Correlation matrices for categorical attributes
    # categorial_correlation_matrix_wrapper()

    # Preprocess data
    return_tuple_avc, return_tuple_salary = preprocess_data_wrapper()
    
    # Parse return tuples
    X_avc_full, T_avc_full, label_encoder_avc, onehot_encoder_avc, X_avc_train, T_avc_train, X_avc_test, T_avc_test = return_tuple_avc
    X_salary_full, T_salary_full, label_encoder_salary, onehot_encoder_salary, X_salary_train, T_salary_train, X_salary_test, T_salary_test = return_tuple_salary

    # Logistic Regression
    return_tuple_avc_logreg, return_tuple_salary_logreg, return_tuple_sklearn_logreg = logistic_regression_wrapper(X_avc_train, T_avc_train, X_avc_test, T_avc_test, X_salary_train, T_salary_train, X_salary_test, T_salary_test)

    # First call for LogReg algorithm
    process_and_generate_reports(
        return_tuple_avc_logreg, return_tuple_salary_logreg, return_tuple_sklearn_logreg,
        X_avc_train, T_avc_train, X_avc_test, T_avc_test, 
        X_salary_train, T_salary_train, X_salary_test, T_salary_test, 
        "LogReg"
    )
        
    # Define the MLP architecture and training parameters
    return_tuple_avc_mlp, return_tuple_salary_mlp, return_tuple_sklearn_mlp = mlp_wrapper(X_avc_train, T_avc_train, X_avc_test, T_avc_test, X_salary_train, T_salary_train, X_salary_test, T_salary_test)
    
    # Second call for MLP algorithm
    process_and_generate_reports(
        return_tuple_avc_mlp, return_tuple_salary_mlp, return_tuple_sklearn_mlp,
        X_avc_train, T_avc_train, X_avc_test, T_avc_test, 
        X_salary_train, T_salary_train, X_salary_test, T_salary_test, 
        "MLP"
    )

__main__()