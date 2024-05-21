import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder

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


# Encode categorical variables
def encode_categorical(df, target_column, encoder=None):
    df = df.copy()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    categorical_columns.remove(target_column)
    
    label_encoder = LabelEncoder()
    df[target_column] = label_encoder.fit_transform(df[target_column])
    
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        df_onehot = pd.DataFrame(encoder.fit_transform(df[categorical_columns]), columns=encoder.get_feature_names_out(categorical_columns))
    else:
        df_onehot = pd.DataFrame(encoder.transform(df[categorical_columns]), columns=encoder.get_feature_names_out(categorical_columns))
    
    df = pd.concat([df, df_onehot], axis=1)
    df.drop(categorical_columns, axis=1, inplace=True)
    
    return df, label_encoder, encoder

# Encode categorical variables and prepare datasets for logistic regression
def preprocess_data(df, target_column, encoder=None):
    df_encoded, label_encoder, encoder = encode_categorical(df, target_column, encoder)
    X = df_encoded.drop(columns=[target_column]).values
    T = df_encoded[target_column].values
    return X, T, label_encoder, encoder