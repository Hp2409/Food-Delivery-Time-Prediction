"""preprocess.py
Generic preprocessing utilities for the ML project.
Exposes: load_data, clean_data, split_data
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path, filename):
    """Load CSV data from data/ folder.

    Args:
        path: path to data folder (default 'data')
        filename: filename in the data folder
    Returns:
        pandas.DataFrame
    """
    full = os.path.join(path, filename)
    return pd.read_csv(full)

def clean_data(df):
    """Basic cleaning: drops fully empty rows, fills numeric NAs with median.
    Customize this for your dataset.
    """
    df = df.dropna(how='all').copy()
    for col in df.select_dtypes(include=['number']).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else '') 
    return df

def split_data(df, target_column, test_size=0.2, random_state=42):
    """Split features and target into train/test sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
