import pandas as pd

def preprocess_data(df):
    # Remove id 
    df.drop('id', axis=1, inplace=True)
    
    df['age'] = df['age'] / 365.25
    
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Filter blood pressure outliers
    df = df[(df['ap_hi'] >= 50) & (df['ap_hi'] <= 250)]
    df = df[(df['ap_lo'] >= 30) & (df['ap_lo'] <= 150)]
    
    return df