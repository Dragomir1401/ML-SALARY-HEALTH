import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder

# Function to identify and impute missing values
def impute_missing_values(df, numeric_columns, categorical_columns, fit=True, imputer=None):
    """Impute missing values for numeric and categorical columns."""
    df_numeric = df[numeric_columns]  # Select numeric columns
    df_categorical = df[categorical_columns]  # Select categorical columns

    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)

    # Explicitly control downcasting behavior
    pd.set_option('future.no_silent_downcasting', True)
    df.replace('not_defined', np.nan, inplace=True)
    pd.set_option('future.no_silent_downcasting', False)

    if fit:
        # Create and fit imputer for numeric columns
        numeric_imputer = SimpleImputer(strategy='mean')
        df_numeric_imputed = pd.DataFrame(numeric_imputer.fit_transform(df_numeric), columns=numeric_columns)
        
        # Create and fit imputer for categorical columns
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df_categorical_imputed = pd.DataFrame(categorical_imputer.fit_transform(df_categorical), columns=categorical_columns)
        
        # Store imputers for train use
        imputers = {'numeric': numeric_imputer, 'categorical': categorical_imputer}
    else:
        # Transform numeric and categorical columns using provided imputers
        df_numeric_imputed = pd.DataFrame(imputer['numeric'].transform(df_numeric), columns=numeric_columns)
        df_categorical_imputed = pd.DataFrame(imputer['categorical'].transform(df_categorical), columns=categorical_columns)
        imputers = imputer

    # Combine imputed numeric and categorical columns
    df_imputed = pd.concat([df_numeric_imputed, df_categorical_imputed], axis=1)
    
    return df_imputed, imputers

# Function to identify and handle extreme values using IQR and impute them
def handle_extreme_values(df, numeric_columns):
    """Handle extreme values in numeric columns using the Interquartile Range (IQR) method and impute them."""
    Q1 = df[numeric_columns].quantile(0.25)  # First quartile (25th percentile)
    Q3 = df[numeric_columns].quantile(0.75)  # Third quartile (75th percentile)
    IQR = Q3 - Q1  # Interquartile Range
    lower_bound = Q1 - 1.5 * IQR  # Lower bound for outliers
    upper_bound = Q3 + 1.5 * IQR  # Upper bound for outliers

    df_outliers_handled = df.copy()  # Create a copy of the dataframe
    for col in numeric_columns:
        # Identify outliers and replace them with NaN
        df_outliers_handled[col] = np.where((df_outliers_handled[col] < lower_bound[col]) | 
                                            (df_outliers_handled[col] > upper_bound[col]), 
                                            np.nan, df_outliers_handled[col])
    
    # Impute missing values (extreme values marked as NaN)
    imputer = SimpleImputer(strategy='mean')
    df_outliers_handled[numeric_columns] = imputer.fit_transform(df_outliers_handled[numeric_columns])
    
    return df_outliers_handled

# Function to remove highly correlated attributes
def remove_redundant_attributes(df, numeric_columns, threshold=0.9):
    """Remove highly correlated attributes based on the specified threshold."""
    numeric_df = df[numeric_columns]  # Select numeric columns
    corr_matrix = numeric_df.corr().abs()  # Compute the absolute correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))  # Select the upper triangle of the correlation matrix
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]  # Identify columns to drop
    df_reduced = df.drop(columns=to_drop)  # Drop the identified columns
    return df_reduced, to_drop

# Function to standardize numerical attributes
def standardize_data(df, numeric_columns, method='standard', fit=True, scaler=None):
    """Standardize numerical attributes using the specified method."""
    if fit:
        # Select the scaling method
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        # Fit and transform the numeric columns
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        return df, scaler
    else:
        # Transform the numeric columns using the provided scaler
        df[numeric_columns] = scaler.transform(df[numeric_columns])
        return df

# Encode categorical variables
def encode_categorical(df, target_column, encoder=None):
    """Encode categorical variables and target column."""
    df = df.copy()  # Create a copy of the dataframe
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()  # Identify categorical columns
    
    # Ensure the target column is not included in categorical columns
    if target_column in categorical_columns:
        categorical_columns.remove(target_column)
    
    # Encode the target column
    label_encoder = LabelEncoder()
    df[target_column] = label_encoder.fit_transform(df[target_column])
    
    # Encode categorical columns using OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    df_onehot = pd.DataFrame(encoder.fit_transform(df[categorical_columns]), columns=encoder.get_feature_names_out(categorical_columns))
    
    # Concatenate the original dataframe with the one-hot encoded dataframe
    df = pd.concat([df, df_onehot], axis=1)
    # Drop the original categorical columns
    df.drop(categorical_columns, axis=1, inplace=True)
    
    return df, label_encoder, encoder

def preprocess_data(df, target_column, encoder=None):
    """Preprocess the data by encoding categorical variables and separating features and target."""
    df_encoded, label_encoder, encoder = encode_categorical(df, target_column, encoder)  # Encode categorical variables
    X = df_encoded.drop(columns=[target_column]).values  # Separate features
    T = df_encoded[target_column].values  # Separate target
    return X, T, label_encoder, encoder
