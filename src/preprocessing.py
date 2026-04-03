import pandas as pd
import numpy as np


def preprocess_data(df):
    """Preprocess cardiovascular disease dataset"""
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty")
    
    required_cols = ['id', 'age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cardio']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove id 
    df = df.copy()
    df.drop('id', axis=1, inplace=True)
    
    # Convert age from days to years
    df['age'] = df['age'] / 365.25
    
    # Calculate BMI
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Filter blood pressure outliers
    df = df[(df['ap_hi'] >= 50) & (df['ap_hi'] <= 250)]
    df = df[(df['ap_lo'] >= 30) & (df['ap_lo'] <= 150)]
    
    # Validate target variable
    if not set(df['cardio'].unique()).issubset({0, 1}):
        raise ValueError("Target variable 'cardio' must contain only 0 and 1")
    
    return df