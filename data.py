import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import os

# Reading the CSV files
avc_df_full = pd.read_csv('tema2_AVC/AVC_full.csv')
avc_df_train = pd.read_csv('tema2_AVC/AVC_train.csv')
avc_df_test = pd.read_csv('tema2_AVC/AVC_test.csv')

salary_df_full = pd.read_csv('tema2_SalaryPrediction/SalaryPrediction_full.csv')
salary_df_train = pd.read_csv('tema2_SalaryPrediction/SalaryPrediction_train.csv')
salary_df_test = pd.read_csv('tema2_SalaryPrediction/SalaryPrediction_test.csv')

def analyze_attributes(df):
    attribute_info = pd.DataFrame(columns=['Attribute', 'Type', 'Number of Missing Values', 'Number of Unique Values'])
    for column in df.columns:
        dtype = df[column].dtype
        num_missing = df[column].isnull().sum()
        unique_values = df[column].nunique()
        attribute_info = attribute_info._append({
            'Attribute': column,
            'Type': dtype,
            'Number of Missing Values': num_missing,
            'Number of Unique Values': unique_values
        }, ignore_index=True)
    return attribute_info

def numeric_statistics(df, numeric_columns):
    statistics = df[numeric_columns].describe(percentiles=[0.25, 0.5, 0.75]).T
    statistics['number_of_non_missing_values'] = df[numeric_columns].notnull().sum()
    statistics = statistics.rename(columns={
        'mean': 'mean_value',
        'std': 'standard_deviation',
        'min': 'minimum_value',
        '25%': '25th_percentile',
        '50%': '50th_percentile',
        '75%': '75th_percentile',
        'max': 'maximum_value'
    })
    return statistics[['number_of_non_missing_values', 'mean_value', 'standard_deviation', 'minimum_value', '25th_percentile', '50th_percentile', '75th_percentile', 'maximum_value']]

def plot_normalized_boxplot(df, numeric_columns, title):
    # Normalize the data
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df[numeric_columns])
    normalized_df = pd.DataFrame(normalized_data, columns=numeric_columns)
    
    # Plot the normalized data using pandas boxplot
    plt.figure(figsize=(15, 10))
    normalized_df.boxplot(rot=45)
    plt.title(title, fontsize=20)
    plt.xlabel('Numeric Columns', fontsize=15)
    plt.ylabel('Normalized Values', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

def categorical_statistics(df, categorical_columns):
    statistics = pd.DataFrame(columns=['Attribute', 'Number of Non-Missing Values', 'Number of Unique Values'])
    for column in categorical_columns:
        num_missing = df[column].notnull().sum()
        unique_values = df[column].nunique()
        statistics = statistics._append({
            'Attribute': column,
            'Number of Non-Missing Values': num_missing,
            'Number of Unique Values': unique_values
        }, ignore_index=True)
    return statistics

def plot_histograms(df, categorical_columns, title):
    plt.figure(figsize=(20, 15))
    num_columns = 3
    num_rows = len(categorical_columns) // num_columns + (len(categorical_columns) % num_columns > 0)
    
    for i, column in enumerate(categorical_columns, 1):
        plt.subplot(num_rows, num_columns, i)
        df[column].value_counts().plot(kind='bar')
        plt.title(column, fontsize=12)
    
    plt.suptitle(title, y=0.92, fontsize=20)  # Adjust the y position of the title
    plt.subplots_adjust(hspace=0.5, top=0.85)  # Increase hspace to add space between rows and adjust top to distance title
    plt.show()

def plot_class_balance(df, class_column, title):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=class_column, data=df)
    plt.title(f'Class Balance - {title}', fontsize=16)
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()


# Function to calculate Pearson correlation and plot heatmap for numerical attributes
def plot_correlation_matrix(df, numeric_columns, title):
    correlation_matrix = df[numeric_columns].corr(method='pearson')
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title(f'Correlation Matrix - {title}', fontsize=16)
    plt.show()

# Function to calculate Cramér's V statistic for categorical attributes
def cramers_v(confusion_matrix):
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k-1, r-1))

def plot_categorical_correlation_matrix(df, categorical_columns, title):
    corr_matrix = np.zeros((len(categorical_columns), len(categorical_columns)))
    
    for i, col1 in enumerate(categorical_columns):
        for j, col2 in enumerate(categorical_columns):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                confusion_matrix = pd.crosstab(df[col1], df[col2])
                corr_matrix[i, j] = cramers_v(confusion_matrix)
    
    corr_df = pd.DataFrame(corr_matrix, index=categorical_columns, columns=categorical_columns)
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title(f'Categorical Correlation Matrix - {title}', fontsize=16)
    plt.show()
    
# Function to identify and impute missing values
def impute_missing_values(df, numeric_columns, categorical_columns):
    # Separate numeric and categorical columns
    df_numeric = df[numeric_columns]
    df_categorical = df[categorical_columns]
    
    # Impute numeric columns
    numeric_imputer = SimpleImputer(strategy='mean')
    df_numeric_imputed = pd.DataFrame(numeric_imputer.fit_transform(df_numeric), columns=numeric_columns)
    
    # Impute categorical columns
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df_categorical_imputed = pd.DataFrame(categorical_imputer.fit_transform(df_categorical), columns=categorical_columns)
    
    # Combine the imputed columns back into a single DataFrame
    df_imputed = pd.concat([df_numeric_imputed, df_categorical_imputed], axis=1)
    
    return df_imputed


# Function to identify and handle extreme values using IQR
def handle_extreme_values(df, numeric_columns):
    Q1 = df[numeric_columns].quantile(0.25)
    Q3 = df[numeric_columns].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_outliers_handled = df.copy()
    for col in numeric_columns:
        df_outliers_handled[col] = np.where((df_outliers_handled[col] < lower_bound[col]) | 
                                            (df_outliers_handled[col] > upper_bound[col]), 
                                            np.nan, df_outliers_handled[col])
    return df_outliers_handled

# Function to remove highly correlated attributes
def remove_redundant_attributes(df, numeric_columns, threshold=0.9):
    # Ensure only numeric columns are considered for correlation
    numeric_df = df[numeric_columns]
    
    # Calculate the correlation matrix
    corr_matrix = numeric_df.corr().abs()
    
    # Select the upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Identify columns to drop based on the threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print("Correlation Matrix:\n", corr_matrix)
    print("Upper Triangle Matrix:\n", upper)
    print("Columns to drop:", to_drop)
    
    # Drop the identified columns from the original DataFrame
    df_reduced = df.drop(columns=to_drop)
    
    return df_reduced, to_drop


# Function to standardize numerical attributes
def standardize_data(df, numeric_columns, method='standard'):
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df

def __main__():
    # Create output directory
    if not os.path.exists('output'):
        os.makedirs('output')

    # Analyze attributes
    avc_attribute_info = analyze_attributes(avc_df_full)
    salary_attribute_info = analyze_attributes(salary_df_full)
    
    # Open 2 files in dir output/
    with open('output/avc_attribute_info.txt', 'w') as f:
        f.write(avc_attribute_info.to_string())
        
    with open('output/salary_attribute_info.txt', 'w') as f:
        f.write(salary_attribute_info.to_string())

    # Numeric statistics
    numeric_columns_avc = ['mean_blood_sugar_level', 'body_mass_indicator', 'years_old', 'analysis_results', 'biological_age_index']
    numeric_columns_salary = ['fnl', 'hpw', 'gain', 'edu_int', 'years', 'loss', 'prod']

    avc_numeric_statistics = numeric_statistics(avc_df_full, numeric_columns_avc)
    salary_numeric_statistics = numeric_statistics(salary_df_full, numeric_columns_salary)

    # Write numeric statistics to files
    with open('output/avc_numeric_statistics.txt', 'w') as f:
        f.write(avc_numeric_statistics.to_string())
        
    with open('output/salary_numeric_statistics.txt', 'w') as f:
        f.write(salary_numeric_statistics.to_string())

    # Boxplot for numeric attributes
    # plot_normalized_boxplot(avc_df_full, numeric_columns_avc, 'AVC - Boxplot for Numeric Attributes')
    # plot_normalized_boxplot(salary_df_full, numeric_columns_salary, 'Salary - Boxplot for Numeric Attributes')

    # Categorical statistics
    categorical_columns_avc = ['cardiovascular_issues', 'job_category', 'sex', 'tobacco_usage', 'high_blood_pressure', 'married', 'living_area', 'chaotic_sleep', 'cerebrovascular_accident']
    categorical_columns_salary = ['relation', 'country', 'job', 'work_type', 'partner', 'edu', 'gender', 'race', 'gtype']

    avc_categorical_statistics = categorical_statistics(avc_df_full, categorical_columns_avc)
    salary_categorical_statistics = categorical_statistics(salary_df_full, categorical_columns_salary)

    # Display categorical statistics into files
    with open('output/avc_categorical_statistics.txt', 'w') as f:
        f.write(avc_categorical_statistics.to_string())
        
    with open('output/salary_categorical_statistics.txt', 'w') as f:
        f.write(salary_categorical_statistics.to_string())

    # Histograms for categorical attributes and save them
    # plot_histograms(avc_df_full, categorical_columns_avc, 'AVC - Histograms for Categorical Attributes')
    # plot_histograms(salary_df_full, categorical_columns_salary, 'Salary - Histograms for Categorical Attributes')
    
    # Define the class columns for each dataset
    class_column_avc = 'cerebrovascular_accident'  # This is the class column in the AVC dataset
    class_column_salary = 'money'  # This is the class column in the Salary dataset

    # Plot class balance using seaborn for each dataset
    # plot_class_balance(avc_df_full, class_column_avc, 'AVC Dataset Full')
    # plot_class_balance(avc_df_train, class_column_avc, 'AVC Dataset Train')
    # plot_class_balance(avc_df_train, class_column_avc, 'AVC Dataset Test')
    # plot_class_balance(salary_df_full, class_column_salary, 'Salary Dataset Full')
    # plot_class_balance(salary_df_train, class_column_salary, 'Salary Dataset Train')
    # plot_class_balance(salary_df_test, class_column_salary, 'Salary Dataset Test')
    
    # # Plot correlation matrices for numerical attributes
    # plot_correlation_matrix(avc_df_full, numeric_columns_avc, 'AVC Dataset Full')
    # plot_correlation_matrix(salary_df_full, numeric_columns_salary, 'Salary Dataset Full')
    # plot_correlation_matrix(avc_df_train, numeric_columns_avc, 'AVC Dataset Train')
    # plot_correlation_matrix(salary_df_train, numeric_columns_salary, 'Salary Dataset Train')
    # plot_correlation_matrix(avc_df_test, numeric_columns_avc, 'AVC Dataset Test')
    # plot_correlation_matrix(salary_df_test, numeric_columns_salary, 'Salary Dataset Test')

    # # Plot correlation matrices for categorical attributes
    # plot_categorical_correlation_matrix(avc_df_full, categorical_columns_avc, 'AVC Dataset Full')
    # plot_categorical_correlation_matrix(salary_df_full, categorical_columns_salary, 'Salary Dataset Full')
    # plot_categorical_correlation_matrix(avc_df_train, categorical_columns_avc, 'AVC Dataset Train')
    # plot_categorical_correlation_matrix(salary_df_train, categorical_columns_salary, 'Salary Dataset Train')
    # plot_categorical_correlation_matrix(avc_df_test, categorical_columns_avc, 'AVC Dataset Test')
    # plot_categorical_correlation_matrix(salary_df_test, categorical_columns_salary, 'Salary Dataset Test')
    
    # Preprocessing for full, train, and test datasets
    datasets = {
        'avc_full': avc_df_full,
        'avc_train': avc_df_train,
        'avc_test': avc_df_test,
        'salary_full': salary_df_full,
        'salary_train': salary_df_train,
        'salary_test': salary_df_test
    }
    
    
    for name, df in datasets.items():
        # Separate columns
        numeric_columns = numeric_columns_avc if 'avc' in name else numeric_columns_salary
        categorical_columns = categorical_columns_avc if 'avc' in name else categorical_columns_salary

        # Impute missing values
        df_imputed = impute_missing_values(df, numeric_columns, categorical_columns)

        # Handle extreme values
        df_outliers_handled = handle_extreme_values(df_imputed, numeric_columns)
        
        # Impute missing values again after handling extreme values
        df_outliers_imputed = impute_missing_values(df_outliers_handled, numeric_columns, categorical_columns)

        # Remove redundant attributes
        df_reduced, dropped_columns = remove_redundant_attributes(df_outliers_imputed, numeric_columns, threshold=0.9)
        
        # Print what columns have been dropped and why
        with open(f'output/{name}_dropped_columns.txt', 'w') as f:
            f.write(f'Dropped columns: {dropped_columns}\n')
            f.write(f'Reason: Highly correlated with other columns\n')

        # Standardize numerical attributes
        df_standardized = standardize_data(df_reduced, numeric_columns, method='standard')

        # Write the processed data to files for testing
        df_standardized.to_csv(f'output/{name}_processed.csv', index=False)
        
__main__()