import pandas as pd
import os


def load_data(filepath):
    """Load csv data"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    if not filepath.endswith('.csv'):
        raise ValueError("File must be CSV format")
    
    df = pd.read_csv(filepath, sep=';')
    
    if df.empty:
        raise ValueError("Dataset is empty")
    
    return df