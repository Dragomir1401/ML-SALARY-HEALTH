import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

# Function to analyze attributes of a DataFrame
def analyze_attributes(df):
    """Analyze attributes of a DataFrame and return a summary DataFrame."""
    attribute_info = pd.DataFrame(columns=['Attribute', 'Type', 'Number of Missing Values', 'Number of Unique Values'])
    for column in df.columns:
        dtype = df[column].dtype  # Determine the data type of the column
        num_missing = df[column].isnull().sum()  # Count the number of missing values
        unique_values = df[column].nunique()  # Count the number of unique values
        # Append the information to the summary DataFrame
        attribute_info = attribute_info._append({
            'Attribute': column,
            'Type': dtype,
            'Number of Missing Values': num_missing,
            'Number of Unique Values': unique_values
        }, ignore_index=True)
    return attribute_info

# Function to calculate statistics for numeric columns
def numeric_statistics(df, numeric_columns):
    """Calculate descriptive statistics for numeric columns of a DataFrame."""
    # Calculate summary statistics
    statistics = df[numeric_columns].describe(percentiles=[0.25, 0.5, 0.75]).T
    # Count the number of non-missing values
    statistics['number_of_non_missing_values'] = df[numeric_columns].notnull().sum()
    # Rename columns for clarity
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

# Function to plot boxplots for numeric columns
def plot_boxplot(df, numeric_columns, title_prefix):
    """Plot boxplots for numeric columns in a DataFrame."""
    for column in numeric_columns:
        plt.figure(figsize=(10, 6))
        df.boxplot(column=column)
        plt.title(f'{title_prefix} - {column}', fontsize=20)
        plt.xlabel(column, fontsize=15)
        plt.ylabel('Values', fontsize=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.show()

# Function to calculate statistics for categorical columns
def categorical_statistics(df, categorical_columns):
    """Calculate statistics for categorical columns of a DataFrame."""
    statistics = pd.DataFrame(columns=['Attribute', 'Number of Non-Missing Values', 'Number of Unique Values'])
    for column in categorical_columns:
        num_missing = df[column].notnull().sum()  # Count non-missing values
        unique_values = df[column].nunique()  # Count unique values
        # Append the information to the statistics DataFrame
        statistics = statistics._append({
            'Attribute': column,
            'Number of Non-Missing Values': num_missing,
            'Number of Unique Values': unique_values
        }, ignore_index=True)
    return statistics

# Function to plot histograms for categorical columns
def plot_histograms(df, categorical_columns, title):
    """Plot histograms for categorical columns in a DataFrame."""
    plt.figure(figsize=(20, 15))
    num_columns = 3  # Number of columns in the subplot grid
    num_rows = len(categorical_columns) // num_columns + (len(categorical_columns) % num_columns > 0)  # Calculate the number of rows
    
    for i, column in enumerate(categorical_columns, 1):
        plt.subplot(num_rows, num_columns, i)
        df[column].value_counts().plot(kind='bar')
        plt.title(column, fontsize=12)
    
    plt.suptitle(title, y=0.92, fontsize=20)  # Set the main title and adjust its position
    plt.subplots_adjust(hspace=0.5, top=0.85)  # Adjust space between rows and distance from the top
    plt.show()

# Function to plot class balance
def plot_class_balance(df, class_column, title):
    """Plot the balance of classes in a DataFrame."""
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
    """Plot the Pearson correlation matrix for numeric columns in a DataFrame."""
    correlation_matrix = df[numeric_columns].corr(method='pearson')
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title(f'Correlation Matrix - {title}', fontsize=16)
    plt.show()

# Function to calculate Cramér's V statistic for categorical attributes
def cramers_v(confusion_matrix):
    """Calculate Cramér's V statistic for measuring association between categorical variables."""
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k-1, r-1))

# Function to plot the categorical correlation matrix
def plot_categorical_correlation_matrix(df, categorical_columns, title):
    """Plot the Cramér's V correlation matrix for categorical columns in a DataFrame."""
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
