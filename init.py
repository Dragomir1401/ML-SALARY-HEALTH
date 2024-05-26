import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE, ADASYN, SMOTEN, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import TomekLinks
from mlp import train_and_evaluate_manual_mlp, train_and_evaluate_sklearn_mlp
from logistic_regression import train_and_eval_logistic, train_and_eval_sklearn_logistic
from evaluation import generate_confusion_matrices_mlp, generate_confusion_matrices_logreg, generate_classification_reports_logreg, generate_classification_reports_mlp, plot_all_learning_curves, generate_comparative_table
from preprocess import impute_missing_values, handle_extreme_values, remove_redundant_attributes, preprocess_data, standardize_data, encode_categorical
from visualisation import analyze_attributes, numeric_statistics, plot_boxplot, categorical_statistics, plot_histograms, plot_class_balance, plot_correlation_matrix, plot_categorical_correlation_matrix

# Reading the CSV files for avc
avc_df_full = pd.read_csv('tema2_AVC/AVC_full.csv')
avc_df_train = pd.read_csv('tema2_AVC/AVC_train.csv')
avc_df_test = pd.read_csv('tema2_AVC/AVC_test.csv')

# Reading the CSV files for salary
salary_df_full = pd.read_csv('tema2_SalaryPrediction/SalaryPrediction_full.csv')
salary_df_train = pd.read_csv('tema2_SalaryPrediction/SalaryPrediction_train.csv')
salary_df_test = pd.read_csv('tema2_SalaryPrediction/SalaryPrediction_test.csv')

# Numeric columns
numeric_columns_avc = ['mean_blood_sugar_level', 'body_mass_indicator', 'years_old', 'analysis_results', 'biological_age_index']
numeric_columns_salary = ['fnl', 'hpw', 'gain', 'edu_int', 'years', 'loss', 'prod']

# Categorical columns
categorical_columns_avc = ['cardiovascular_issues', 'job_category', 'sex', 'tobacco_usage', 'high_blood_pressure', 'married', 'living_area', 'chaotic_sleep', 'cerebrovascular_accident']
categorical_columns_salary = ['relation', 'country', 'job', 'work_type', 'partner', 'edu', 'gender', 'race', 'gtype', 'money']

# Define the class columns for each dataset
class_column_avc = 'cerebrovascular_accident'
class_column_salary = 'money'

# Preprocessing for full, train, and test datasets
datasets = {
    'avc_train': avc_df_train,
    'salary_train': salary_df_train,
    'avc_full': avc_df_full,
    'salary_full': salary_df_full,
    'avc_test': avc_df_test,
    'salary_test': salary_df_test
}

# Wrapper functions for analysis of the datasets
def analyze_wrapper():
    # Analyze attributes
    avc_attribute_info = analyze_attributes(avc_df_full)
    salary_attribute_info = analyze_attributes(salary_df_full)
    
    # Write the attribute information to files
    with open('output/avc_attribute_info.txt', 'w') as f:
        f.write(avc_attribute_info.to_string())
        
    with open('output/salary_attribute_info.txt', 'w') as f:
        f.write(salary_attribute_info.to_string())

# Wrapper functions for numeric statistics
def numeric_statistics_wrapper():
    avc_numeric_statistics = numeric_statistics(avc_df_full, numeric_columns_avc)
    salary_numeric_statistics = numeric_statistics(salary_df_full, numeric_columns_salary)

    # Write numeric statistics to files
    with open('output/avc_numeric_statistics.txt', 'w') as f:
        f.write(avc_numeric_statistics.to_string())
        
    with open('output/salary_numeric_statistics.txt', 'w') as f:
        f.write(salary_numeric_statistics.to_string())
    
# Wrapper functions for boxplots    
def boxplots_wrapper(dataset_avc, dataset_salary):
    # Boxplot for numeric attributes
    plot_boxplot(dataset_avc, numeric_columns_avc, 'AVC - Boxplot for Numeric Attributes')
    plot_boxplot(dataset_salary, numeric_columns_salary, 'Salary - Boxplot for Numeric Attributes')

# Wrapper functions for categorical statistics
def categorial_statistics_wrapper():
    avc_categorical_statistics = categorical_statistics(avc_df_full, categorical_columns_avc)
    salary_categorical_statistics = categorical_statistics(salary_df_full, categorical_columns_salary)
    
    # Display categorical statistics into files
    with open('output/avc_categorical_statistics.txt', 'w') as f:
        f.write(avc_categorical_statistics.to_string())
        
    with open('output/salary_categorical_statistics.txt', 'w') as f:
        f.write(salary_categorical_statistics.to_string())
        
# Wrapper functions for histograms
def histograms_wrapper():
    # Histograms for categorical attributes and save them
    plot_histograms(avc_df_full, categorical_columns_avc, 'AVC - Histograms for Categorical Attributes')
    plot_histograms(salary_df_full, categorical_columns_salary, 'Salary - Histograms for Categorical Attributes')

# Wrapper functions for class balance
def class_balance_wrapper():
    # Plot class balance using seaborn for each dataset
    plot_class_balance(avc_df_full, class_column_avc, 'AVC Dataset Full')
    plot_class_balance(avc_df_train, class_column_avc, 'AVC Dataset Train')
    plot_class_balance(avc_df_train, class_column_avc, 'AVC Dataset Test')
    plot_class_balance(salary_df_full, class_column_salary, 'Salary Dataset Full')
    plot_class_balance(salary_df_train, class_column_salary, 'Salary Dataset Train')
    plot_class_balance(salary_df_test, class_column_salary, 'Salary Dataset Test')

# Wrapper functions for correlation matrices
def correlation_matrix_wrapper():
    # Plot correlation matrices for numerical attributes
    plot_correlation_matrix(avc_df_full, numeric_columns_avc, 'AVC Dataset Full')
    plot_correlation_matrix(salary_df_full, numeric_columns_salary, 'Salary Dataset Full')
    plot_correlation_matrix(avc_df_train, numeric_columns_avc, 'AVC Dataset Train')
    plot_correlation_matrix(salary_df_train, numeric_columns_salary, 'Salary Dataset Train')
    plot_correlation_matrix(avc_df_test, numeric_columns_avc, 'AVC Dataset Test')
    plot_correlation_matrix(salary_df_test, numeric_columns_salary, 'Salary Dataset Test')

# Wrapper functions for categorical correlation matrices
def categorial_correlation_matrix_wrapper():
    # Plot correlation matrices for categorical attributes
    plot_categorical_correlation_matrix(avc_df_full, categorical_columns_avc, 'AVC Dataset Full')
    plot_categorical_correlation_matrix(salary_df_full, categorical_columns_salary, 'Salary Dataset Full')
    plot_categorical_correlation_matrix(avc_df_train, categorical_columns_avc, 'AVC Dataset Train')
    plot_categorical_correlation_matrix(salary_df_train, categorical_columns_salary, 'Salary Dataset Train')
    plot_categorical_correlation_matrix(avc_df_test, categorical_columns_avc, 'AVC Dataset Test')
    plot_categorical_correlation_matrix(salary_df_test, categorical_columns_salary, 'Salary Dataset Test')

# Wrapper functions for preprocessing the datasets 
def preprocess_data_wrapper():
    processed_datasets = {}
    fitted_scalers = {}
    fitted_imputers = {}
    fitted_encoders = {}

    for name, df in datasets.items():
        # Skip the full datasets
        if 'full' in name:
            continue

        print(f"Processing dataset: {name}")

        # Determine the target column based on dataset name
        target_column = class_column_avc if 'avc' in name else class_column_salary
        if target_column not in df.columns:
            raise KeyError(f"Target column '{target_column}' not found in dataset '{name}'")

        # Separate numeric and categorical columns
        numeric_columns = numeric_columns_avc if 'avc' in name else numeric_columns_salary
        categorical_columns = categorical_columns_avc if 'avc' in name else categorical_columns_salary

        if 'train' in name:
            # Eliminate highly correlated attributes
            df_reduced, dropped_columns = remove_redundant_attributes(df, numeric_columns, threshold=0.5)
            print(f"Dropped columns in {name}: {dropped_columns}")

            # Adjust numeric columns after dropping correlated ones
            numeric_columns_after_drop = [col for col in numeric_columns if col not in dropped_columns]

            # Impute missing values
            df_imputed, imputer = impute_missing_values(df_reduced, numeric_columns_after_drop, categorical_columns, fit=True)
            fitted_imputers[name] = imputer

            # Handle extreme values and impute again
            df_outliers_handled = handle_extreme_values(df_imputed, numeric_columns_after_drop)

            # Standardize numerical attributes
            df_standardized, scaler = standardize_data(df_outliers_handled, numeric_columns_after_drop, method='standard', fit=True)
            fitted_scalers[name] = scaler

            # Encode categorical variables
            df_encoded, label_encoder, onehot_encoder = encode_categorical(df_standardized, target_column, encoder=None)
            fitted_encoders[name] = (label_encoder, onehot_encoder)
            
            # Apply a oversampler to balance the classes
            oversampler = SVMSMOTE()
            X_oversampled, T_oversampled = oversampler.fit_resample(df_encoded.drop(columns=[target_column]), df_encoded[target_column])
            df_encoded = pd.DataFrame(X_oversampled, columns=df_encoded.drop(columns=[target_column]).columns)
            df_encoded[target_column] = T_oversampled
            
            # Apply Tomek Links to balance the classes
            # tl = TomekLinks()
            # X_resampled, y_resampled = tl.fit_resample(df_encoded.drop(columns=[target_column]), df_encoded[target_column])
            # df_encoded = pd.DataFrame(X_resampled, columns=df_encoded.drop(columns=[target_column]).columns)
            # df_encoded[target_column] = y_resampled

            processed_datasets[name] = df_encoded
            processed_datasets[name].to_csv(f'output/{name}_processed.csv', index=False)

        else:
            # Get the name for the inputers used on train
            train_name = name.replace('test', 'train').replace('full', 'train')
            dropped_columns = [col for col in numeric_columns if col not in processed_datasets[train_name].columns]
            numeric_columns_after_drop = [col for col in numeric_columns if col not in dropped_columns]

            # Get the fitted objects from the train data
            imputer = fitted_imputers[train_name]
            scaler = fitted_scalers[train_name]

            # Apply the transformations using the fitted objects
            df_imputed, _ = impute_missing_values(df, numeric_columns_after_drop, categorical_columns, fit=False, imputer=imputer)
            df_outliers_handled = handle_extreme_values(df_imputed, numeric_columns_after_drop)
            df_standardized = standardize_data(df_outliers_handled, numeric_columns_after_drop, method='standard', fit=False, scaler=scaler)
            df_encoded, _, _ = encode_categorical(df_standardized, target_column, encoder=None)

            # Ensure the test data has the same columns as the train data
            train_columns = processed_datasets[train_name].columns
            for col in train_columns:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
            df_encoded = df_encoded[train_columns]

            # Write the processed data to a CSV file
            processed_datasets[name] = df_encoded
            processed_datasets[name].to_csv(f'output/{name}_processed.csv', index=False)

    # Prepare datasets for learning algorithms
    X_avc_train, T_avc_train, _, _ = preprocess_data(processed_datasets['avc_train'], class_column_avc, encoder=None)
    X_avc_test, T_avc_test, _, _ = preprocess_data(processed_datasets['avc_test'], class_column_avc, encoder=None)

    X_salary_train, T_salary_train, _, _ = preprocess_data(processed_datasets['salary_train'], class_column_salary, encoder=None)
    X_salary_test, T_salary_test, _, _ = preprocess_data(processed_datasets['salary_test'], class_column_salary, encoder=None)

    return_tuple_avc = (X_avc_train, T_avc_train, X_avc_test, T_avc_test)
    return_tuple_salary = (X_salary_train, T_salary_train, X_salary_test, T_salary_test)

    return return_tuple_avc, return_tuple_salary


def logistic_regression_wrapper(X_avc_train, T_avc_train, X_avc_test, T_avc_test, X_salary_train, T_salary_train, X_salary_test, T_salary_test):
    # Manual Logistic Regression
    w_avc, train_nll_avc, test_nll_avc, train_acc_avc, test_acc_avc = train_and_eval_logistic(
        X_avc_train, T_avc_train, X_avc_test, T_avc_test, lr=0.3, epochs_no=1000)
    w_salary, train_nll_salary, test_nll_salary, train_acc_salary, test_acc_salary = train_and_eval_logistic(
        X_salary_train, T_salary_train, X_salary_test, T_salary_test, lr=0.3, epochs_no=1000)

    # Logistic Regression using scikit-learn
    model_avc, _, _, train_acc_avc_sklearn, test_acc_avc_sklearn = train_and_eval_sklearn_logistic(
        X_avc_train, T_avc_train, X_avc_test, T_avc_test, penalty='l1', C=0.01, solver='liblinear')
    model_salary, _, _, train_acc_salary_sklearn, test_acc_salary_sklearn = train_and_eval_sklearn_logistic(
        X_salary_train, T_salary_train, X_salary_test, T_salary_test, penalty='l2', C=10, solver='liblinear')

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
    # Define the MLP architecture and training parameters for the manual MLP
    input_size_avc = X_avc_train.shape[1]
    input_size_salary = X_salary_train.shape[1]
    output_size = 2  # Binary classification

    # Hyperparameters for AVC dataset found by GridSearchCV
    hidden_layer_sizes_avc = (100, 50, 25)
    learning_rate_avc = 0.0001
    l2_reg_avc = 0.0001
    epochs_avc = 2000
    batch_size_avc = 32
    optimizer_avc = 'SGD'
    
    # Hyperparameters for Salary dataset found by GridSearchCV
    hidden_layer_sizes_salary = (100, 50, 25)
    learning_rate_salary = 0.001
    l2_reg_salary = 0.01
    epochs_salary = 1500
    batch_size_salary = 32
    optimizer_salary = 'Adam'

    # Manual MLP Training and Evaluation for AVC dataset
    mlp_manual_avc, train_acc_avc_manual, test_acc_avc_manual, train_loss_avc_manual, test_loss_avc_manual = train_and_evaluate_manual_mlp(
        X_avc_train, T_avc_train, X_avc_test, T_avc_test, input_size_avc, hidden_layer_sizes_avc[0], output_size, epochs_avc, learning_rate_avc, l2_reg_avc, batch_size_avc, optimizer_avc
    )
    
    # Manual MLP Training and Evaluation for Salary dataset
    mlp_manual_salary, train_acc_salary_manual, test_acc_salary_manual, train_loss_salary_manual, test_loss_salary_manual = train_and_evaluate_manual_mlp(
        X_salary_train, T_salary_train, X_salary_test, T_salary_test, input_size_salary, hidden_layer_sizes_salary[0], output_size, epochs_salary, learning_rate_salary, l2_reg_salary, batch_size_salary, optimizer_salary
    )

    # Scikit-learn MLP Training and Evaluation for AVC dataset
    train_acc_avc_sklearn, test_acc_avc_sklearn, model_avc_sklearn = train_and_evaluate_sklearn_mlp(
        X_avc_train, T_avc_train, X_avc_test, T_avc_test, hidden_layer_sizes_avc, epochs_avc, learning_rate_avc, l2_reg_avc
    )

    # Scikit-learn MLP Training and Evaluation for Salary dataset
    train_acc_salary_sklearn, test_acc_salary_sklearn, model_salary_sklearn = train_and_evaluate_sklearn_mlp(
        X_salary_train, T_salary_train, X_salary_test, T_salary_test, hidden_layer_sizes_salary, epochs_salary, learning_rate_salary, l2_reg_salary
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
    
def find_best_logreg_hyperparams(X_train, T_train):
    # Define the parameter grid
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }

    # Initialize the logistic regression model
    log_reg = LogisticRegression()

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=5, scoring='f1', verbose=1, n_jobs=-1)

    # Fit GridSearchCV on the training data
    grid_search.fit(X_train, T_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Return the best model and the report
    return best_model

def find_best_mlp_hyperparams(X_train, T_train):
    # Define the parameter grid
    param_grid = {
        'hidden_layer_sizes': [(100,), (100, 50), (100, 50, 25)],
        'max_iter': [1000, 1500, 2000],  # Increase the number of iterations
        'learning_rate_init': [0.01, 0.001, 0.0001],
        'alpha': [0.0001, 0.001, 0.01],
        'solver': ['adam', 'sgd'],  # Try different solvers
        'early_stopping': [True]  # Enable early stopping
    }

    # Initialize the MLP model
    mlp = MLPClassifier()

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5, scoring='f1', verbose=1, n_jobs=-1)

    # Fit GridSearchCV on the training data
    grid_search.fit(X_train, T_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Return the best model and the report
    return best_model

def __main__():
    # Create output directory
    if not os.path.exists('output'):
        os.makedirs('output')
        
    # # Analyze attributes
    # analyze_wrapper()
    
    # # Numeric statistics
    # numeric_statistics_wrapper()

    # # Boxplots for numeric attributes
    # boxplots_wrapper(avc_df_full, salary_df_full)

    # # Categorial statistics
    # categorial_statistics_wrapper()

    # # Histograms for categorical attributes
    # histograms_wrapper()
    
    # # Class balance
    # class_balance_wrapper()
    
    # Correlation matrices for numerical attributes
    # correlation_matrix_wrapper()
    
    # # Correlation matrices for categorical attributes
    # categorial_correlation_matrix_wrapper()

    # Preprocess data
    return_tuple_avc, return_tuple_salary = preprocess_data_wrapper()
    
    # boxplots_wrapper(avc_df_train, salary_df_train)
    
    # Parse return tuples
    X_avc_train, T_avc_train, X_avc_test, T_avc_test = return_tuple_avc
    X_salary_train, T_salary_train, X_salary_test, T_salary_test = return_tuple_salary
    
    # # Find best hyperparameters for logistic regression
    # best_model_avc = find_best_logreg_hyperparams(X_avc_train, T_avc_train)
    # best_model_salary = find_best_logreg_hyperparams(X_salary_train, T_salary_train)
    
    # # Print best hyperparameters
    # print(f"Best hyperparameters for AVC dataset: {best_model_avc.get_params()}")
    # print(f"Best hyperparameters for Salary dataset: {best_model_salary.get_params()}")
    
    # # Find best hyperparameters for MLP
    # best_model_avc_mlp = find_best_mlp_hyperparams(X_avc_train, T_avc_train)
    # best_model_salary_mlp = find_best_mlp_hyperparams(X_salary_train, T_salary_train)
    
    # # Print best hyperparameters
    # print(f"Best hyperparameters for MLP on AVC dataset: {best_model_avc_mlp.get_params()}")
    # print(f"Best hyperparameters for MLP on Salary dataset: {best_model_salary_mlp.get_params()}")
    

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