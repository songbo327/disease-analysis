import pandas as pd

def load_data(filepath):
    """Load csv data"""
    df = pd.read_csv(filepath, sep=';')
    return df